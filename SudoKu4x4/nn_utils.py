import SudokuMaster

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.distributions.categorical import Categorical
import torchvision
import torch.optim as optimizer
from torchvision import datasets
from tqdm.auto import tqdm
import time

from pathlib import Path
# from collections import namedtuple
# from sudoku import Sudoku

import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = 4
num_classes = 5

def eval_pred(pred):
    sat = SudokuMaster.checkBoardValidity(pred.tolist())
    return sat

def eval_sudoku(preds, labels):
    single_correct = (preds == labels.cpu().numpy()).reshape(-1)
    total_correct = []
    for pred in preds:
        total_correct.append(int(SudokuMaster.checkBoardValidity(pred.tolist())))
    total_correct = np.array(total_correct)
    return single_correct, total_correct

def save(net, file_name, epoch=0):
    state = {
            'net': net,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + file_name + '_' + str(epoch) + '.t7'
    torch.save(state, save_point)
    return net

def compute_rewards(preds):
    rewards = []
    for pred in preds:
        rewards.append(eval_pred(pred))
    rewards = [1.0 if x else 0. for x in rewards]
    return np.array(rewards)

def evaluate(model, dataloader):
    model.eval() 
    
    board_corrects = []
    single_corrects = []

    for sample in dataloader:
        img_seq = sample['input']
        label_seq = sample['label']
        img_seq = img_seq.cuda()
        label_seq = label_seq.cuda()

        N, M1, M2, C, H, W = img_seq.shape
        x = img_seq.reshape(N*M1*M2, C, H, W).cuda()            
        batch_logits = model(x).reshape(N, M1, M2, -1)
        masked_probs = batch_logits.reshape(N, M1, M2, -1)
        selected_probs, preds = torch.max(masked_probs, -1)
        preds = preds.data.cpu().numpy().reshape(N, M1, M2)
        single_correct, total_correct = eval_sudoku(preds, label_seq)
        single_corrects.extend(single_correct)
        board_corrects.extend(total_correct)

    acc = np.mean(single_corrects)
    board_acc = np.mean(board_corrects)
    
    return acc, board_acc

# def walk(batch_logits, batch_labels, T):
#     criterion = nn.CrossEntropyLoss()
#     accepts = 0
#     loss = 0
#     for idx, ite in enumerate(zip(batch_logits, batch_labels)):
#         x, label = ite
#         low = 1; up = 5
#         orig_label = np.random.randint(low=low, high=up)
#         pseudos = label.clone()
#         index = torch.where(pseudos <= 5)
#         pseudos[pseudos == orig_label] == 10
#         pseudo_label = np.random.randint(low=low, high=up)
#         # remove old labels and solve for new label
#         pseudos[pseudos <= 5] = 0
#         pseudos[pseudos == 10] = pseudo_label
#         sat, sol = check(pseudos)
#         if sat == False:
#             continue
#         pseudos = torch.Tensor(sol).long()
#         with torch.no_grad():
#             origin_loss = criterion(x.reshape(-1, num_classes), label.reshape(-1).cuda()).item()
#             new_loss = criterion(x.reshape(-1, num_classes), pseudos.reshape(-1).cuda()).item()
#         if new_loss < origin_loss or T == 0.0:
#             batch_labels[idx, :, :] = pseudos
#             accepts += 1 # accept
#             loss += new_loss
#         elif np.exp((origin_loss-new_loss)/T) >= np.random.rand():
#             batch_labels[idx, :, :] = pseudos
#             accepts += 1 # accept
#             loss += new_loss
#         else:
#             loss += origin_loss
#             continue
#     return batch_labels, accepts