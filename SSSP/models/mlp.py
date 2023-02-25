from nn_utils import *

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc = nn.Linear(128, len(sym_list))

    def forward(self, x):
        x = self.fc(x)
        return x