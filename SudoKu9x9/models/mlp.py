import sys 
sys.path.append("..") 
from nn_utils import *
import torch.nn.functional as F

# class LeNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)      
#         self.fc1   = nn.Linear(16*5*5, 120)
#         self.fc2   = nn.Linear(120, 84)       
#         self.linear = nn.Linear(84, num_classes)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv2(out))
#         out = F.max_pool2d(out, 2)      
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = self.linear(out)
#         return out

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)      
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)       
        self.linear = nn.Linear(84, num_classes)

    def forward(self, x, dropout=False):
        if dropout == True:
            self.dropout = nn.Dropout(0.25)
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)      
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.dropout(out)
            out = self.linear(out)
        else:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)      
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.linear(out)
        return out