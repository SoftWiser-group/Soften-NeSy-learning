from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
import os
from PIL import Image
import json
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import time

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

op_list = ['+', '-', '*', '/']
digit_list = [str(i) for i in range(1, 10)] 
sym_list =  ['UNK'] + digit_list + op_list
def sym2id(sym):
    return sym_list.index(sym)
def id2sym(idx):
    return sym_list[idx]

unk_idx = sym2id('UNK')
digit_idx_list = [sym2id(x) for x in digit_list]
op_idx_list = [sym2id(x) for x in op_list]


root_dir = './'
img_dir = root_dir + 'data/Handwritten_Math_Symbols/'
img_size = 45

def equal_res(preds, gts):
    return (np.abs(preds - gts)) < 1e-2

res_precision = 5

def eval_expr(preds, seq_len):
    res_preds = []
    expr_preds = []
    for i_pred, i_len in zip(preds, seq_len):
        i_pred = i_pred[:i_len]
        i_expr = ''.join([id2sym(idx) for idx in i_pred])
        try:
            i_res_pred = np.float(eval(i_expr))
        except:
            i_res_pred = np.inf
        res_preds.append(i_res_pred)
        expr_preds.append(i_expr)
    return expr_preds, res_preds

def eval_pred(i_pred, i_len):
    i_pred = i_pred[:i_len]
    i_expr = ''.join([id2sym(idx) for idx in i_pred])
    try:
        i_res_pred = np.float(eval(i_expr))
    except:
        i_res_pred = np.inf
    return i_expr, i_res_pred

def compute_rewards(preds, res, seq_len):
    expr_preds, res_preds = eval_expr(preds, seq_len)
    rewards = equal_res(res_preds, res)
    rewards = [1.0 if x else 0. for x in rewards]
    return np.array(rewards)

def save(net, file_name, epoch=0):
    state = {
            'net': net,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '.t7'
    torch.save(state, save_point)
    return net

    
def evaluate(model, dataloader):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    for sample in dataloader:
        img_seq = sample['img_seq']
        label_seq = sample['label_seq']
        res = sample['res']
        seq_len = sample['len']
        expr = sample['expr']
        img_seq = img_seq.to(device)
        label_seq = label_seq.to(device)

#         masked_probs = model(img_seq)
#         selected_probs, preds = torch.max(masked_probs, -1)
#         selected_probs = torch.log(selected_probs+1e-12)
#         expr_preds, res_preds = eval_expr(preds.data.cpu().numpy(), seq_len)

        N, M, C, H, W = img_seq.shape
        x = img_seq.reshape(N*M, C, H, W).cuda()            
        batch_logits = model(x).reshape(N, M, -1)
        masked_probs = batch_logits.reshape(N, M, -1)
        selected_probs, preds = torch.max(masked_probs, -1)
        # selected_probs = torch.log(selected_probs+1e-12)
        expr_preds, res_preds = eval_expr(preds.data.cpu().numpy(), seq_len)
        
        res_pred_all.append(res_preds)
        res_all.append(res)
        expr_pred_all.extend(expr_preds)
        expr_all.extend(expr)
        

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    # print('Grammar Error: %.2f'%(np.isinf(res_pred_all).mean()*100))
    acc = equal_res(res_pred_all, res_all).mean()

    
    expr_pred_all = ''.join(expr_pred_all)
    expr_all = ''.join(expr_all)
    sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
    
    return acc, sym_acc

def Semantic_loss(x, y, num_classes=14):
    n = len(x)
    y_ = F.one_hot(y, num_classes)
    x_ = F.log_softmax(x, dim=-1)
    loss = -(x_*y_ + torch.clamp(1-x_, min=-4.6)*(1-y_)).sum() / n # log(0.01)=-4.6
    return loss

# some global info
projection = [0, 1, 3, 5, 6]
drop = [2, 4]

# def walk(batch_logits, batch_labels, batch_results, T):
#     criterion = nn.CrossEntropyLoss()
#     accepts = 0
#     loss = 0
#     for idx, ite in enumerate(zip(batch_logits, batch_labels, batch_results)):
#         x, label, res = ite
#         ind = np.random.randint(len(projection))
#         if ind in {1, 2, 3}:
#             low = 10; up = 14
#         elif ind in {0, 4}:
#             low = 1; up = 10

#         pseudo_label = torch.Tensor([np.random.randint(low=low, high=up)]).long().cuda()
#         pseudos = label.clone()
#         pseudos[projection[ind]] = pseudo_label
#         sat, sol = check(pseudos[projection].tolist(), res.item())
#         if sat == False:
#             continue
#         pseudos[drop] = torch.Tensor(sol).long()
#         with torch.no_grad():
#             origin_loss = criterion(x, label.cuda()).item()
#             new_loss = criterion(x, pseudos.cuda()).item()
#         if new_loss < origin_loss or T == 0.0:
#             batch_labels[idx, :] = pseudos
#             accepts += 1 # accept
#             loss += new_loss
#         elif np.exp((origin_loss-new_loss)/T) >= np.random.rand():
#             batch_labels[idx, :] = pseudos
#             accepts += 1 # accept
#             loss += new_loss
#         else:
#             loss += origin_loss
#             continue
#     return batch_labels, accepts



