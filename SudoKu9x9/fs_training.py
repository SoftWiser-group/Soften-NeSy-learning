import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from nn_utils import *
from dataset import *

import sys 
sys.path.append("..") 
import models
from utils import *

from smt_solver import init_check, check
from joblib import Parallel, delayed

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch SudoKu Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--batch_size', default=16, type=float, help='the size of min-batch')
parser.add_argument('--cooling_strategy', default='log', type=str, choices=['log', 'exp', 'linear'], help='cooling schedule')
parser.add_argument('--T', type=float, default=1.0, help='temperature gamma')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
num_epochs = 10000
batch_size = opt.batch_size
sgd_lr = 0.1; # learning rate of SGD
adam_lr = 0.001;  # learning rate of Adam
num_classes = 10; 
sampling_epoch = 20; 
# setting projection and initilize labels
proj_index = {'1': [(0,0), (3,3)], '2': [(0,3), (3,6)], '3': [(0,6), (3,9)], \
              '4': [(3,0), (6,3)], '5': [(3,3), (6,6)], '6': [(3,6), (6,9)], \
              '7': [(6,0), (9,3)], '8': [(6,3), (9,6)], '9': [(6,6), (9,9)]}
projection = {'1': [[(0,0),(9,3)],[(0,0),(3,9)]], '2': [[(0,3),(9,6)],[(0,0),(3,9)]], '3': [[(0,6),(9,9)],[(0,0),(3,9)]], \
              '4': [[(0,0),(9,3)],[(3,0),(6,9)]], '5': [[(0,3),(9,6)],[(3,0),(6,9)]], '6': [[(0,6),(9,9)],[(3,0),(6,9)]], \
              '7': [[(0,0),(9,3)],[(6,0),(9,9)]], '8': [[(0,3),(9,6)],[(6,0),(9,9)]], '9': [[(0,6),(9,9)],[(6,0),(9,9)]] }

train_set = SudoKuDataset('train') 
test_set = SudoKuDataset('test')

### generate ground-truth labels
tmp_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                        shuffle=False, num_workers=2)

#### generate random labels
labels = []
for (idx, data) in enumerate(tmp_dataloader):
    X = data['input']
    Y  = torch.zeros_like(data['label'])
    for (i, y) in enumerate(Y):
        print(idx, i)
        sat, sol = init_check(y.numpy())
        sol = torch.Tensor(sol)
        # sol[t == 0] = 0
        labels.append(sol.unsqueeze(dim=0))
labels = torch.cat(labels, dim=0)
torch.save(labels, './data/random_labels.pt')

pseudo_labels = torch.load('./data/random_labels.pt').long()
print('label initialization complete')

def walk(batch_logits, batch_labels, T):
    energy = nn.CrossEntropyLoss(reduction='sum')
    max_iter = len(batch_logits)
    low = 1; up = 9 
    random_ind_pools = np.random.randint(low=low, high=up+1, size=max_iter)
    def walk_(idx):
        loss = 0
        accept = 0
        x = batch_logits[idx]
        label = batch_labels[idx]
        ind = random_ind_pools[idx]
        change = proj_index[str(ind)]
        u1, v1 = change[0]; u2, v2 = change[1]
        pseudos = label.clone()
        tmp = pseudos[u1:u2, v1:v2].clone().reshape(-1).cuda()
        pred = torch.softmax(x[u1:u2, v1:v2,:].clone().reshape(-1, num_classes), dim=-1)
        weights = (F.one_hot(tmp, num_classes)*pred).sum(dim=-1)
        # z = torch.multinomial(input = (-weights/100000.0).exp(), num_samples=2) # uniform sampling
        z = torch.multinomial(input = (-weights).exp(), num_samples=2) # weighted sampling
        z1 = z[0]; z2 = z[1]
        val = tmp[z1].item()
        tmp[z1] = tmp[z2]
        tmp[z2] = val
        pseudos[u1:u2, v1:v2] = tmp.reshape(3,3)
        with torch.no_grad():
            tmp_x = x.reshape(-1, num_classes)
            tmp_label = label.reshape(-1)
            tmp_pseudo = pseudos.reshape(-1)
            origin_loss = energy(tmp_x, tmp_label.cuda()).item()
            new_loss = energy(tmp_x, tmp_pseudo.cuda()).item()
        if new_loss < origin_loss or T == 0.0:
            accept = 1 # accept
            return pseudos.unsqueeze(dim=0)
        elif np.exp((origin_loss-new_loss)/T) >= np.random.rand():
            accept = 1 # accept
            return pseudos.unsqueeze(dim=0)
        else:
            return label.unsqueeze(dim=0)
    pseudos = Parallel(n_jobs=10)(delayed(walk_)(idx) for idx in range(max_iter))
    pseudos = torch.cat(pseudos, dim=0)
    accepts = (batch_labels.reshape(-1, 9*9) != pseudos.reshape(-1, 9*9)).any(dim=-1).sum()
    return pseudos, accepts


def train(net, train_set, test_set, opt):
    # temperature setting
    T=T0=opt.T; t=1; T_min=0.1; update_T = False; 

    writer = SummaryWriter(comment=opt.exp_name + '_' + 'fs' + '_' + opt.cooling_strategy)
    best_acc = 0.0

    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=True, num_workers=2)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2)

    optim_cls = optimizer.SGD(net.parameters(), lr=sgd_lr)
    criterion = nn.CrossEntropyLoss()

    count = 0
    for epoch in range(num_epochs):
        pred_acc = 0
        gt_acc = 0
        train_acc = []
        train_loss = 0
        total_changes = 0
        net.train()
        for batch_idx, sample in enumerate(train_dataloader):
            img_seq = sample['input']
            batch_truths = sample['label'] # shall not be used
            index = sample['index']

            N, M1, M2, C, H, W = img_seq.shape
            x = img_seq.reshape(N*M1*M2, C, H, W).cuda()            
            batch_logits = net(x).reshape(N, M1, M2, -1)
            x = batch_logits.reshape(-1, num_classes)

            if epoch <= sampling_epoch: # initilization: train on uniform sampling
                batch_labels = pseudo_labels[index, :, :]
                pseudos, changes = walk(batch_logits, batch_labels, 0.0)
                total_changes += changes
                pseudo_labels[index, :, :] = pseudos
                y = pseudos.reshape(-1).cuda()       
                cls_loss = criterion(x, y)
                optim_cls.zero_grad()
                cls_loss.backward()
                optim_cls.step()
            elif epoch > sampling_epoch and epoch % sampling_epoch == 0: # update net parameters for each sampling epoch
                batch_labels = pseudo_labels[index, :, :]
                y = batch_labels.reshape(-1).cuda()        
                cls_loss = criterion(x, y)
                optim_cls.zero_grad()
                cls_loss.backward()
                optim_cls.step()
                update_T = True
            else: # update y
                batch_labels = pseudo_labels[index, :, :]
                pseudos, changes = walk(batch_logits, batch_labels, T)
                total_changes += changes
                pseudo_labels[index, :, :] = pseudos
                y = pseudos.reshape(-1).cuda()        
                cls_loss = criterion(x, y)

            train_loss += cls_loss.item()
        
            _, pred = torch.max(x, dim=-1)
            pred_acc += (pred == y).float().mean().item()

            gt_labels = batch_truths.reshape(-1)
            gt_acc += (gt_labels == y.cpu()).float().mean().item()
                
            sys.stdout.write('\r')
            ite = (batch_idx+1)
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.3f \t total_changes: %3d \t acc/gtacc %.3f/%.3f \t T %.4f'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), \
                    train_loss/ite, total_changes, pred_acc/ite, gt_acc/ite, T))
            sys.stdout.flush()   

            ## eval for each batch
            _, preds = torch.max(batch_logits, -1)
            preds = preds.data.cpu().numpy().reshape(N, M1, M2)
            single_correct, total_correct = eval_sudoku(preds, batch_truths)
            acc = np.mean(single_correct)
            board_acc = np.mean(total_correct)
            acc = 100*float(acc)
            total_acc = 100*float(board_acc)
            train_acc.append(board_acc)

            count += 1
            writer.add_scalars('train_accs', {'pred_acc': pred_acc/ite*100.0, 'gt_acc': gt_acc/ite*100.0, 'sym_acc': acc, 'board_acc': total_acc}, count)

        ## save best based on training set
        if np.mean(train_acc) > best_acc:
            best_acc = np.mean(train_acc)
            save(net, file_name+'_best')

        # update temperature T
        if update_T == True:
            if opt.cooling_strategy == 'log':
                T = T0 / np.log(1+t)
                t += 1
            elif opt.cooling_strategy == 'exp':
                dT = 0.95
                T = T * dT
            elif opt.cooling_strategy == 'linear':
                dT = 0.01 * 1.0 / np.sqrt(t)
                T = T - dT
            T = max(T_min, T)
            update_T = False

        acc, board_acc = evaluate(net, eval_dataloader)
        acc = 100*float(acc)
        board_acc = 100*float(board_acc)
        print('\n \t Test | Epoch %3d Acc@1: %2f%% Board_acc: %.2f%%' % (epoch, acc, board_acc))
        writer.add_scalars('val_accs', {'acc': acc, 'board_acc': board_acc}, epoch)

    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'lenet' + '_' + str(opt.seed) + '_' + opt.exp_name + '_' + opt.cooling_strategy
    net = models.LeNet(num_classes)
    net.cuda()
    net = train(net, train_set, test_set, opt)
    save(net, file_name)
