from dataset import *

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import sys 
sys.path.append("..") 
import models
from utils import *
from smt_solver import sample_search

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch HWF Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--batch_size', default=64, type=float, help='the size of min-batch')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

num_epochs = 100; 
adam_lr = 1e-3; 
sgd_lr = 0.01; 

train_set = ShortPathDataset('train') 
test_set = ShortPathDataset('test')
# pesudo_labels = torch.load('./data/random_labels.pt')

# tmp_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
#                         shuffle=False, num_workers=2, collate_fn=Graph_collate)
## generate random labels
# labels = []
# for batch_idx, graph in enumerate(train_graphs):
#     print(batch_idx)
#     sat, label = init_check(graph)
#     if sat == True:
#         labels.append(torch.Tensor(label).unsqueeze(dim=0))
#     else:
#         print('error')
#         break
# labels = torch.cat(labels, dim=0)
# torch.save(labels, './data/random_labels.pt')

def train(net, train_set, test_set, opt):

    writer = SummaryWriter(comment=opt.exp_name + '_' + '')

    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=True, num_workers=8, collate_fn=Graph_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=8, collate_fn=Graph_collate)

    optim = optimizer.Adam(net.parameters(), lr=adam_lr)
    # optim = optimizer.SGD(net.parameters(), lr=sgd_lr)

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        train_loss = 0
        gt_corr = 0
        net.train()
        for batch_idx, sample in enumerate(train_dataloader):
            batch_inputs = sample['input']
            adj_matrix = sample['adj']
            res = sample['label'] # shall not be used
            index = sample['index']
            sol = sample['path']

            m = batch_inputs.shape[0]
            x = batch_inputs.reshape(m, n, n).cuda()
            x = net(x).reshape(m, n).cuda()
            y = res.cuda()     
            loss = criterion(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()


            corrs = []
            for p, r in zip(y.data.cpu().numpy(), res.data.cpu().numpy()):
                corr, pval = stats.spearmanr(p, r)
                corrs.append(corr)
            gt_corr += np.array(corrs).mean()
            
            train_loss += loss.item()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t loss: %.5f gt_corr: %.2f'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), train_loss/(batch_idx+1), gt_corr/(batch_idx+1)))
            sys.stdout.flush()   

        acc, corr = evaluate(net, train_dataloader)
        acc = 100*float(acc)
        print('\n \t Train | Epoch %3d Acc: %2f%% Correlation: %.2f' % (epoch, acc, corr))
        writer.add_scalars('train_accs', {'acc': acc, 'correlation': corr}, epoch)

        # # eval for each epoch
        acc, corr = evaluate(net, eval_dataloader)
        acc = 100*float(acc)
        print('\n \t Test | Epoch %3d Acc: %2f%% Correlation: %.2f' % (epoch, acc, corr))
        writer.add_scalars('val_accs', {'acc': acc, 'correlation': corr}, epoch)
    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'vae_net' + '_' + str(opt.seed) + '_' + opt.exp_name
    # net = torch.load('./checkpoint/vae_net_0_vns_log_3000.t7')['net']
    module = models.VGAE()
    classifier = models.MLPNet()
    net = models.CatNet(module, classifier)
    net.cuda()
    # pesudo_labels = torch.load('./data/pesudo_labels.pt')

    net = train(net, train_set, test_set, opt)
    save(net, file_name)