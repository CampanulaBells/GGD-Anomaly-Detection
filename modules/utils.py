import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
from utils import load_mat, preprocess_features

def draw_line(plt, epoch_list, data_list, color, label, max_marker=None):
    plt.plot(epoch_list, data_list, color=color, label=label)
    if max_marker is not None:
        max_index = np.argmax(data_list)
        val_str = "{:.4f}".format(data_list[max_index])
        plt.plot(epoch_list[max_index], data_list[max_index], color=color, marker=max_marker)
        plt.annotate(val_str, xytext=(-20, 10), textcoords='offset points',
                     xy=(epoch_list[max_index], data_list[max_index]), color=color)

def load_dataset(dataset):
    print('Dataset: {}'.format(dataset), flush=True)
    adj, features, _, _, _, _, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)
    features, _ = preprocess_features(features)
    src, dst = np.nonzero(adj)
    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)
    return g, features, ano_label, str_ano_label, attr_ano_label

def set_random_seeds(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)