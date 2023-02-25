from nn_utils import *
import random
import torch
import copy

class ShortPathDataset(Dataset):
    def __init__(self, split='train', numSamples=None, randomSeed=None, aug=False):
        super(ShortPathDataset, self).__init__()
        
        self.split = split
        self.dataset = torch.load('./data/%s_graphs.pt'%split)
        
        if numSamples:
            self.dataset = self.dataset[:numSamples]

        self.img_transform = transforms.Compose([
                           transforms.ToTensor()])
                        #    transforms.Normalize((0.5,), (1,))])
        self.aug = aug
    
    def __getitem__(self, index):
        sample = { 'input': None, 'label': None, 
                    'index': None, 'path': None}
        data = copy.deepcopy(self.dataset[index])
        x = torch.Tensor(data.input)
        x[x == 0] = max_weight+1
        x = 1 - (x.float() / (max_weight+1))
        sample['adj'] = torch.Tensor(x)
        sample['input'] = torch.Tensor(x)
        sample['label'] = torch.Tensor(data.d[0,:])
        sample['index'] = index
        sample['path'] = data.paths
        return sample
            
    def __len__(self):
        return len(self.dataset)

def Graph_collate(batch):
    temp = { 'input': None, 'label': None, 
                'index': None, 'path': None}
    temp['adj'] = default_collate([sample['adj'] for sample in batch])
    temp['input'] = default_collate([sample['input'] for sample in batch])
    temp['label'] = default_collate([sample['label'] for sample in batch])
    temp['index'] = default_collate([sample['index'] for sample in batch])
    temp['path'] = [sample['path'] for sample in batch]
    return temp
