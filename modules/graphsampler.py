import dgl
import torch


class GraphSampler(torch.nn.Module):
    def __init__(self, graph, n_nodes=4):
        super(GraphSampler, self).__init__()
        self.g = graph

        indices_list = []
        probs_list = []

        A = self.g.adjacency_matrix()
        for i in range(self.g.num_nodes()):
            row = A[i]._indices()
            size = row.size()[1]
            tensor_ones = torch.ones(size)
            indices = torch.cat([tensor_ones.unsqueeze(0) * i, row], dim=0)
            indices_list.append(indices)
            if size < n_nodes:
                probs_list.append(tensor_ones)
            else:
                dropout_prob = n_nodes / size
                probs_list.append(tensor_ones * dropout_prob)
        probs = torch.cat(probs_list)
        self.sampler = torch.distributions.bernoulli.Bernoulli(probs=probs.to(self.g.device))
        self.indices = torch.cat(indices_list, dim=1).to(torch.int32).to(self.g.device)

    def sample(self):
        new_indices = torch.masked_select(
            self.indices,
            self.sampler.sample().to(torch.bool)
        ).reshape((2, -1))
        graph = dgl.graph((new_indices[0, :], new_indices[1, :]), num_nodes=self.g.num_nodes())
        if 'feat' in self.g.ndata:
            graph.ndata['feat'] = self.g.ndata['feat']
        if 'label' in self.g.ndata:
            graph.ndata['label'] = self.g.ndata['label']
        return dgl.add_self_loop(graph)

