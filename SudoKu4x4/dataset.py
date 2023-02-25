from nn_utils import *
import random
import torch
import copy

class SudoKuDataset(Dataset):
    def __init__(self, split='train', numSamples=None, randomSeed=None):
        super(SudoKuDataset, self).__init__()
        
        self.split = split
        X_train, Y_train = torch.load('./data/%sset.pt'%split)
        self.dataset = TensorDataset(X_train, Y_train)
        
        if numSamples:
            self.dataset = self.dataset[:numSamples]
    
    def __getitem__(self, index):
        sample = { 'input': None, 'label': None, 'index': None}
        # data = copy.deepcopy(self.dataset[index])
        X, Y = self.dataset[index]
        sample['input'] = X
        sample['label'] = Y
        sample['index'] = index
        return sample
            
    def __len__(self):
        return len(self.dataset)

def Sudoku_collate(batch):
    batch = default_collate(batch)
    return batch
