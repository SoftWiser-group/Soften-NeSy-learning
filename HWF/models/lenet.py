import sys 
sys.path.append("..") 
from nn_utils import *

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(30976, 128)
#         self.fc2 = nn.Linear(128, len(sym_list))

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

class LeNet(nn.Module):
    def __init__(self, num_classes=14):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*8*8, 120)
        self.fc2   = nn.Linear(120, 84)
        self.linear = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.linear(out)
        return out

# class LeNet(nn.Module):
#     def __init__(self, num_classes=14):
#         super(LeNet, self).__init__()
#         self.fc1 = nn.Linear(45*45, num_classes)

#     def forward(self, x):
#         out = x.view(x.size(0), -1)
#         out = self.fc1(out)
#         return out