import torch
import torch.nn as nn
from gcn import GCN
from dgl.nn.pytorch import SGConv


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(n_hidden, n_hidden, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features, summary):
        s = self.bilinear(features, summary)
        return s


class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k=1):
        super(Encoder, self).__init__()
        self.g = g
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        elif gnn_encoder == 'sgc':
            self.conv = SGConv(in_feats, n_hidden, k=10, cached=True)

    def forward(self, features):
        if self.gnn_encoder == 'gcn':
            features = self.conv(features)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(self.g, features)
        return features


class GraphLocalGraphPooling(nn.Module):
    def __init__(self, g, n_hop):
        # TODO: Simulate random walk (randomly drop some subgraph)
        super(GraphLocalGraphPooling, self).__init__()
        A = g.adjacency_matrix().to_dense()
        A = A + torch.eye(A.shape[0])
        A_n = A
        for i in range(n_hop):
            A_n = torch.matmul(A_n, A)
        # TODO: Check matrix situation (sym, factor
        A = torch.sign(A_n)
        self.A = torch.matmul(torch.diag(1 / torch.sum(A, dim=1)), A)
        self.A = self.A.cuda()

    def forward(self, feature):
        # feature: [n_nodes, n_features]
        feature = torch.matmul(self.A, feature)
        return feature


class Model(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation, gnn_encoder, subgraph_size):
        super(Model, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, 1, activation, 0.0, gnn_encoder)
        self.discriminator = Discriminator(n_hidden)
        if subgraph_size > 0:
            self.graph_average_pooling = GraphLocalGraphPooling(g, subgraph_size)
        else:
            self.graph_average_pooling = lambda x: x
        self.graph_conv_layers = self.encoder.conv.layers
        self.dropout = torch.nn.Dropout(0.2)
        # GGD
        self.lin = nn.Linear(n_hidden, n_hidden)

    def forward(self, features):
        features = self.dropout(features)
        embedding_node = features
        for i, graph_conv_layer in enumerate(self.graph_conv_layers):
            embedding_node = graph_conv_layer._activation(
                torch.matmul(embedding_node, graph_conv_layer.weight) + graph_conv_layer.bias)

        embedding_graph_pos = self.encoder(features)
        # avg pooling
        embedding_graph_readout = self.graph_average_pooling(embedding_node)
        # Add skip connection
        embedding_graph_proj = (embedding_graph_pos + embedding_graph_readout) / 2
        # Positive branch of Anomaly
        predicted_score_pos = self.discriminator(embedding_node, embedding_graph_proj)
        # change shape from [n_nodes, 1] to [1, n_nodes]
        predicted_score_pos = torch.swapaxes(predicted_score_pos, 0, 1)

        # Negative branch of Anomaly
        perm = torch.randperm(embedding_node.shape[0])
        embedding_node_neg = embedding_node[perm]
        predicted_score_neg = self.discriminator(embedding_node_neg, embedding_graph_proj)
        predicted_score_neg = torch.swapaxes(predicted_score_neg, 0, 1)

        # ggd
        ggd_score_pos = self.lin(embedding_graph_proj).sum(1).unsqueeze(0)

        embedding_graph_neg = self.encoder(features[perm])
        ggd_score_neg = self.lin(embedding_graph_neg).sum(1).unsqueeze(0)

        return predicted_score_pos, predicted_score_neg, ggd_score_pos, ggd_score_neg, perm


class GCNAdj(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """

    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCNAdj, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.matmul(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out[0])


class ModelGCN(nn.Module):
    def __init__(self, g, A, in_feats, n_hidden, activation, gnn_encoder, subgraph_size):
        super(ModelGCN, self).__init__()
        self.A = A
        self.encoder = GCNAdj(in_feats, n_hidden, activation)

        self.discriminator = Discriminator(n_hidden)
        if subgraph_size > 0:
            self.graph_average_pooling = GraphLocalGraphPooling(g, subgraph_size)
        else:
            self.graph_average_pooling = lambda x: x
        self.dropout = torch.nn.Dropout(0.2)
        # GGD
        self.lin = nn.Linear(n_hidden, n_hidden)

    def forward(self, features):
        features = self.dropout(features)
        embedding_node = self.encoder.act(self.encoder.fc(features) + self.encoder.bias)
        embedding_graph_pos = self.encoder(features, self.A)
        # avg pooling
        embedding_graph_readout = self.graph_average_pooling(embedding_node)
        # Add skip connection
        embedding_graph_proj = (embedding_graph_pos + embedding_graph_readout) / 2
        # Positive branch of Anomaly
        predicted_score_pos = self.discriminator(embedding_node, embedding_graph_proj)
        # change shape from [n_nodes, 1] to [1, n_nodes]
        predicted_score_pos = torch.swapaxes(predicted_score_pos, 0, 1)

        # Negative branch of Anomaly
        perm = torch.randperm(embedding_node.shape[0])
        embedding_node_neg = embedding_node[perm]
        predicted_score_neg = self.discriminator(embedding_node_neg, embedding_graph_proj)
        predicted_score_neg = torch.swapaxes(predicted_score_neg, 0, 1)

        # ggd
        ggd_score_pos = self.lin(embedding_graph_proj).sum(1).unsqueeze(0)

        embedding_graph_neg = self.encoder(features[perm], self.A)
        ggd_score_neg = self.lin(embedding_graph_neg).sum(1).unsqueeze(0)

        return predicted_score_pos, predicted_score_neg, ggd_score_pos, ggd_score_neg, perm
