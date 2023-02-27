from dataset import *

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import sys 
sys.path.append("..") 
import models
from utils import *
import scipy.sparse as sp

import time

from smt_solver import init_check, check
from joblib import Parallel, delayed

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch SSSP Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--batch_size', default=64, type=float, help='the size of min-batch')
parser.add_argument('--cooling_strategy', default='log', type=str, choices=['log', 'exp', 'linear'], help='cooling schedule')
parser.add_argument('--T', type=float, default=1.0, help='temperature gamma')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
num_epochs = 500
batch_size = opt.batch_size
sgd_lr = 0.1; # learning rate of MLP
adam_lr = 0.001;  # learning rate of VAE
sampling_epoch = 10; 

# setting projection and initilize labels
drop = [6*i for i in range(5)]
projection = [i for i in range(30) if i not in drop]


train_set = ShortPathDataset('train') 
test_set = ShortPathDataset('test')

# save gt paths
train_graphs = torch.load('./data/train_graphs.pt')
test_graphs = torch.load('./data/test_graphs.pt')

gts = []
for batch_idx, graph in enumerate(train_graphs):
    gts.append(graph.paths)
torch.save(gts, './data/train_paths.pt')
gts = []
for batch_idx, graph in enumerate(test_graphs):
    gts.append(graph.paths)
torch.save(gts, './data/test_paths.pt')

tmp_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                        shuffle=False, num_workers=2, collate_fn=Graph_collate)
#### generate random labels
labels = []
for batch_idx, graph in enumerate(train_graphs):
    for i in range(6):
        r = 5 - i
        print(batch_idx, r)
        sat, label = init_check(graph, batch_idx, radius = r)
        if sat == True:
            labels.append(torch.Tensor(label).unsqueeze(dim=0))
            break
        else:
            continue
labels = torch.cat(labels, dim=0)
torch.save(labels, './data/random_labels.pt')

pseudo_labels = torch.load('./data/random_labels.pt')
train_paths = torch.load('./data/train_paths.pt')
test_paths = torch.load('./data/test_paths.pt')
# pesudo_labels = labels.clone()
print('label initialization complete')

def walk(batch_logits, batch_labels, batch_indices, T):
    criterion = nn.MSELoss()
    max_iter = len(batch_logits)
    low = -5; up = 5
    random_index_pools = np.random.randint(low=0, high=len(projection), size=max_iter)
    random_val_pools = np.random.randint(low=low, high=up+1, size=max_iter)
    def walk_(idx):
        loss = 0
        accept = 0
        x = batch_logits[idx]
        label = batch_labels[idx]
        index = batch_indices[idx]
        perturb = torch.Tensor([random_val_pools[idx]]).long().cuda()
        ind = random_index_pools[idx]
        pseudos = label.clone()
        pseudo_label = torch.clamp(pseudos[projection[ind]] + perturb, min=0)
        pseudos[projection[ind]] = pseudo_label
        sat, sol = check(pseudos[projection].long().tolist(), index)
        if sat == False:
            return label.unsqueeze(dim=0)
        pseudos = torch.Tensor(sol).float()
        with torch.no_grad():
            origin_loss = criterion(x, label.cuda()).item()
            new_loss = criterion(x, pseudos.cuda()).item()
        if new_loss < origin_loss or T == 0.0:
            accept = 1 # accept
            return pseudos.unsqueeze(dim=0)
        elif np.exp((origin_loss-new_loss)/T) >= np.random.rand():
            accept = 1 # accept
            return pseudos.unsqueeze(dim=0)
        else:
            return label.unsqueeze(dim=0)
    pseudos = Parallel(n_jobs=15)(delayed(walk_)(idx) for idx in range(max_iter))
    pseudos = torch.cat(pseudos, dim=0)
    accepts = (batch_labels.reshape(-1, n) != pseudos.reshape(-1, n)).any(dim=-1).sum()
    return pseudos, accepts

def train(model, train_set, test_set, opt):
    # temperature setting
    T=T0=opt.T; t=1; T_min=0.02; update_T = False; 

    writer = SummaryWriter(comment=opt.exp_name + '_' + 'fs' + '_' + opt.cooling_strategy)
    best_acc = 0.0
    
    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=True, num_workers=2, collate_fn=Graph_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2, collate_fn=Graph_collate)

    optim_cls = optimizer.Adam(net.parameters(), lr=adam_lr)
    # optim_cls = optimizer.SGD(net.parameters(), lr=sgd_lr)

    criterion = nn.MSELoss()

    count = 0
    for epoch in range(num_epochs):
        pred_acc = 0
        gt_acc = 0
        train_acc = []
        train_loss = 0
        gt_corr = 0
        total_changes = 0
        net.train()
        for batch_idx, sample in enumerate(train_dataloader):
            batch_inputs = sample['input']
            res = sample['label'] # shall not be used
            batch_indices = sample['index']
            batch_results = [train_paths[i] for i in batch_indices]

            x = batch_inputs.cuda()         
            batch_logits = net(x)
            x = batch_logits.reshape(-1, n)

            if epoch <= sampling_epoch: # initilization: train on uniform sampling
                batch_labels = pseudo_labels[batch_indices, :]
                pseudos, changes = walk(batch_logits, batch_labels, batch_indices, 0.0)
                total_changes += changes
                pseudo_labels[batch_indices, :] = pseudos
                y = batch_labels.cuda()     
                cls_loss = criterion(x, y)
                optim_cls.zero_grad()
                cls_loss.backward()
                optim_cls.step()
            elif epoch > sampling_epoch and epoch % sampling_epoch == 0: # update net parameters for each sampling epoch
                batch_labels = pseudo_labels[batch_indices, :]
                y = batch_labels.cuda()           
                cls_loss = criterion(x, y)
                optim_cls.zero_grad()
                cls_loss.backward()
                optim_cls.step()
                update_T = True
            else: # update y
                batch_labels = pseudo_labels[batch_indices, :]
                pseudos, changes = walk(batch_logits, batch_labels, batch_indices, T)
                total_changes += changes
                pseudo_labels[batch_indices, :] = pseudos
                y = batch_labels.cuda()        
                cls_loss = criterion(x, y)

            train_loss += cls_loss.item()

            tmp = 0
            for p, r in zip(y.data.cpu().numpy(), res.data.cpu().numpy()):
                corr, pval = stats.spearmanr(p, r)
                tmp += corr
            gt_corr += tmp/batch_size
                
            sys.stdout.write('\r')
            ite = batch_idx+1
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.3f gt_corr: %.3f \t total_changes: %3d \t T %.3f'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), train_loss/ite, gt_corr/ite, total_changes, T))
            sys.stdout.flush()   

            # eval for each batch
            # preds = batch_logits.clone()
            # path_preds, dist_pred = eval_path(batch_inputs, preds.data.cpu())
            # correct, count = equal_res(path_preds, batch_results)
            # acc = 100*float(correct/count)
            # corrs = []
            # for p, r in zip(dist_pred, res.data.cpu().numpy()):
            #     corr, pval = stats.spearmanr(p, r)
            #     corrs.append(corr)
            # corr = np.array(corrs).mean()
            # train_acc.append(acc)

            # count += 1
            # writer.add_scalars('train_accs', {'acc': acc, 'correlation': corr}, count)

        # save best based on training set
        # if np.mean(train_acc) > best_acc:
            # best_acc = np.mean(train_acc)
            # save(net, file_name+'_best')

        # update temperature T
        if update_T == True:
            if opt.cooling_strategy == 'log':
                T0 = opt.T
                T = T0 / np.log(1+t)
                t += 1
            elif opt.cooling_strategy == 'exp':
                dT = 0.95
                T = T * dT
                t += 1
            elif opt.cooling_strategy == 'linear':
                dT = 0.05 * 1.0 / np.sqrt(t)
                T = T - dT
                t += 1
            T = max(T_min, T)
            update_T = False

        # eval for each epoch
        if epoch % 100 == 0:
            acc, corr = evaluate(net, eval_dataloader)
            acc = 100*float(acc)
            print('\n \t Test | Epoch %3d Acc: %2f%% corr: %.2f' % (epoch, acc, corr))
            writer.add_scalars('val_accs', {'acc': acc, 'corr': corr}, epoch)
    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'mlp' + '_' + str(opt.seed) + '_' + opt.exp_name + '_' + opt.cooling_strategy
    module = models.VGAE()
    classifier = models.MLPNet()
    net = models.CatNet(module, classifier)
    net.cuda()
    net = train(net, train_set, test_set, opt)
    save(net, file_name)
    torch.save(pseudo_labels, 'data/pseudo_labels.pt')
