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
from z3 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = 9
num_classes = 10

def eval_pred(pred):
    sat = SudokuMaster.checkBoardValidity(pred.tolist())
    return sat

def eval_sudoku(preds, labels):
    # re-direct
    # preds_clone = preds.copy()
    # for i in range(1, 10):
    #     index = np.where(labels.cpu().numpy() == i)
    #     vals, counts = np.unique(preds[index].reshape(-1), return_counts=True)
    #     ind = np.argmax(counts)
    #     preds[preds_clone == vals[ind]] = i

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

def evaluate_with_smt(model, dataloader):
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
        probs, preds = torch.max(masked_probs, -1)
        preds = preds.data.cpu().numpy().reshape(N, M1, M2)
        single_correct, total_correct = eval_sudoku(preds, label_seq)
        single_corrects.extend(single_correct)
        board_corrects.extend(total_correct)

    acc = np.mean(single_corrects)
    board_acc = np.mean(board_corrects)
    
    return acc, board_acc

def build_groundtruth(size=9):
    block_size = int(np.sqrt(size))
    gt = torch.zeros(size, size, size**2)
    for i in range(size):
        for j in range(size):
            tmp = torch.zeros(size, size) # assign constraints
            tmp[i, :] = 1
            tmp[:, j] = 1
            bx, by = int(i/block_size), int(j/block_size)
            tmp[bx*block_size:(bx+1)*block_size, by*block_size:(by+1)*block_size] = 1
            tmp[i, j] = 0
            gt[i, j, :] = tmp.reshape(-1)
    return gt.reshape(size**2, size**2)