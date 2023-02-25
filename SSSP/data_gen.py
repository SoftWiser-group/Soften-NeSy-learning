from graphs import Graph

import sys
sys.path.append('pygcn/pygcn')
import numpy as np
import torch

np.random.seed(42)

n = 30 # number of vertices
max_edge = 100
n_train = 3000
n_test = 1000

print('Generating train set...')
train_graphs, test_graphs = [], []
for it in range(n_train):
    print(it)
    m = np.random.randint(n-1, max_edge)
    train_graphs.append(Graph.gen_random_graph(n, m))

print('Generating test set...')
for it in range(n_test):
    print(it)
    m = np.random.randint(n-1, max_edge)
    test_graphs.append(Graph.gen_random_graph(n, m))

torch.save(train_graphs, './data/train_graphs.pt')
torch.save(test_graphs, './data/test_graphs.pt')
print('save complete')
