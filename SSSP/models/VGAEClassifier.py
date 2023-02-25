import sys 
sys.path.append("..") 
from nn_utils import *
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


class VGAE(nn.Module):
    def __init__(self, z_dim=128, im_shape=n):
        super(VGAE, self).__init__()
        output_dim = (im_shape) * (im_shape+1) // 2
        self.encode1 = nn.Linear(im_shape*im_shape, 128)
        # self.encode2 = nn.Linear(1024, 512)
        # self.encode_mu = nn.Linear(512, z_dim) 
        # self.encode_var = nn.Linear(512, z_dim) 

        # self.decode_1 = nn.Linear(z_dim, 512)
        # self.decode_2 = nn.Linear(512, output_dim) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        # self.pos_weight = torch.Tensor([float(n*n - 100) / 100])
        # self.bce = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor(self.pos_weight))

        # for m in self.modules():
            # if isinstance(m, nn.Linear):
                # m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def inference(self, x):
        x = x.view(-1, n*n)
        x = self.encode1(x)
        x = self.relu(x)
        return x


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # self.fc = nn.Linear(128, n, bias=False)
        self.fc = nn.Linear(128, n)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return x


class CatNet(nn.Module):
    def __init__(self, module, classifier):
        # net is vae's encoder
        # classifier is linear classifier
        super(CatNet, self).__init__()
        self.module = module
        self.classifier = classifier

    def forward(self, x):
        x = self.module.inference(x)
        x = self.classifier(x)
        return x
