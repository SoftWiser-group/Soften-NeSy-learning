import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal
import os
from PIL import Image
import json
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import time
from scipy import stats
import spsolve
import astar

n = 30 # number of vertices
max_edge = 100
max_weight = 9.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def equal_res(preds, gts):
    correct = 0.0
    count = 0.0
    for (pred, gt) in zip(preds, gts):
        for i in range(n):
            if pred[str(i)] in gt[str(i)]:
                correct += 1
            count += 1
    return correct, count

# tranform input to original adj
def transform(x):
    x = ((1-x)*(max_weight+1)+1e-3).int()
    x[x == (max_weight+1)] = 0
    return x

def swap(mats, i, j):
    mats = mats.clone()
    for idx, (x,) in enumerate(zip(mats)):
        x[[i,j], :] = x[[j,i], :]
        x[:, [i,j]] = x[:, [j,i]]
        mats[idx] = x
    return mats

def compute_rewards(preds, adjs, res, t=True):
    path_preds, dist_pred = eval_path(adjs.data.cpu(), preds)
    rewards = []
    for (pred, gt) in zip(path_preds, res):
        correct = 0
        for i in range(n):
            if pred[str(i)] in gt[str(i)]:
                correct += 1
        rewards.append(correct)
    # rewards = [1.0 if x else 0. for x in rewards]
    return np.array(rewards)

def eval_path(adjs, preds, t=True):
    if t == True:
        adjs = transform(adjs)
    # adjs = adjs.squeeze(dim=1)
    paths, dists = spsolve.ceval_path(adjs.numpy(), preds.numpy(), n)
    return paths, dists

# def eval_path(adjs, preds, t=True):
#     paths = []
#     dists = []
#     for idx, (adj, pred) in enumerate(zip(adjs, preds)):
#         path = {str(i): None for i in range(n)}
#         def neighbors(n):
#             index = np.where(adj[n,:] != 0)[0]
#             for i in index:
#                 yield i

#         def distance(n1, n2):
#             return adj[n1, n2]

#         def cost(n, goal):
#             return pred[n]

#         for k in range(n):
#             p = list(astar.find_path(k, 0, neighbors_fnct=neighbors,
#                         heuristic_cost_estimate_fnct=cost, distance_between_fnct=distance))
#             path[str(k)] = p
#         paths.append(path)
#     return paths, None

# preds: estimated distance from i to 0
# Note: this adjs is normalized, see dataset.py
def eval_pred(adj, pred):
    dist = []
    path = {str(i): None for i in range(n)}
    for i in range(n):
        j = i
        d = 0
        p = [i] # path
        v = [i] # visited node
        while j != 0:
            index = np.where(adj[j,:] != 0)[0]
            index = list(set(index) - set(v))
            if len(index) > 1:
                k = np.argmin(adj[j, index]+pred[index])
            elif len(index) == 1:
                k = 0
            else:
                break
            d += adj[j, index[k]]
            j = index[k]
            p.append(j)
            v.append(j)
        dist.append(d)
        path[str(i)] = p
    return path, dist

def evaluate(model, dataloader):
    model.eval() 
    corrects = []
    counts = []
    corrs = []

    for batch_idx, sample in enumerate(dataloader):
        inp = sample['input']
        res = sample['label']
        adj = sample['adj']
        index = sample['index']
        results = sample['path']
        inp = inp.to(device)
        res = res.to(device)

        preds = model(inp).reshape(-1, n)
        path_preds, dist_pred = eval_path(adj.data.cpu(), preds.data.cpu())
        correct, count = equal_res(path_preds, results)
        corrects.append(float(correct))
        counts.append(float(count))
        for p, r in zip(preds.data.cpu().numpy(), res.data.cpu().numpy()):
            corr, pval = stats.spearmanr(p, r)
            corrs.append(corr)
        
    acc = np.array(corrects).sum() / np.array(counts).sum()
    corr = np.array(corrs).mean()
    return acc, corr

def save(net, file_name, epoch=0):
    state = {
            'net': net,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '.t7'
    torch.save(state, save_point)
    return net

if __name__ == "__main__":
    t = torch.Tensor(graph.input).int()
    x = t.clone()
    x[x == 0] = max_weight+1
    y = 1 - (x.float() / (max_weight+1))
    
    u = ((1-y)*(max_weight+1) + 1e-10).int()
    z = transform(y)
    index = np.where(t != z)
    print(t[index], x[index], y[index], u[index], v[index])