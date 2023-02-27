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

from joblib import Parallel, delayed
from smt_solver import sample_search, check, init_check

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch HWF Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--batch_size', default=64, type=float, help='the size of min-batch')
parser.add_argument('--cooling_strategy', default='log', type=str, choices=['log', 'exp', 'linear'], help='cooling schedule')
parser.add_argument('--T', type=float, default=2.0, help='temperature gamma')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
num_epochs = 1000
batch_size = opt.batch_size
sgd_lr = 0.1; # learning rate of SGD
adam_lr = 0.001;  # learning rate of Adam
num_classes = 14; len_seq = 7
sampling_epoch = 10; 
# setting projection and initilize labels
projection = [0, 1, 3, 5, 6]
drop = [2, 4]

train_set = MathExprDataset('train', numSamples=int(10000*opt.data_used), randomSeed=777) # if u change the random seed, you must re-generate the random label
test_set = MathExprDataset('test')

#### generate ground-truth labels
tmp_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                        shuffle=False,
                        num_workers=2, collate_fn=MathExpr_collate)

#### generate random labels
results = []
for batch_idx, sample in enumerate(tmp_dataloader):
    results.append(sample['res'])
results = torch.cat(results, dim=0)
labels = torch.zeros(size=(len(train_set),7)).long()
num_list = [0,2,4,6]
op_list = [1,3,5]
labels[:, num_list] = torch.randint(1, 10, size=(len(train_set),4))
labels[:, op_list] = torch.randint(10, 14, size=(len(train_set),3))
for idx, ite in enumerate(zip(labels, results)):
    print(idx)
    label, res = ite
    tmp = label[op_list]
    sat, sol = init_check(tmp.tolist(), res.item())
    while sat == False:
        tmp = torch.randint(10, 14, size=(3,))
        sat, sol = init_check(tmp.tolist(), res.item())
        labels[idx, op_list] = tmp
    labels[idx, num_list] = torch.Tensor(sol).long()
torch.save(labels, './data/random_labels.pt')

pseudo_labels = torch.load('./data/random_labels.pt')
# pseudo_labels = labels.clone()
print('label initialization complete')

def walk(batch_logits, batch_labels, batch_results, T):
    criterion = nn.CrossEntropyLoss()
    max_iter = len(batch_logits)
    random_index_pools = np.random.randint(low=0, high=len(projection), size=max_iter)
    random_value_pools = []
    for ind in random_index_pools:
        if ind in {1, 2, 3}:
            low = 10; up = 14
        elif ind in {0, 4}:
            low = 1; up = 10
        random_value_pools.append(np.random.randint(low=low, high=up))
    def walk_(idx):
        ind = random_index_pools[idx]
        x = batch_logits[idx]; label = batch_labels[idx]; res = batch_results[idx]
        pseudo_label = torch.Tensor([random_value_pools[idx]]).long().cuda()
        pseudos = label.clone()
        pseudos[projection[ind]] = pseudo_label
        sat, sol = check(pseudos[projection].tolist(), res.item())
        if sat == False:
            return label
        pseudos[drop] = torch.Tensor(sol).long()
        with torch.no_grad():
            origin_loss = criterion(x, label.cuda()).item()
            new_loss = criterion(x, pseudos.cuda()).item()
        if new_loss < origin_loss or T == 0.0:
            accept = 1 # accept
            return pseudos
        elif np.exp((origin_loss-new_loss)/T) >= np.random.rand():
            accept = 1 # accept
            return pseudos
        else:
            return label
    pseudo_labels = Parallel(n_jobs=9)(delayed(walk_)(idx) for idx in range(max_iter))
    pseudo_labels = torch.vstack(pseudo_labels)
    accepts = (batch_labels != pseudo_labels).any(dim=-1).sum()
    return pseudo_labels, accepts

def train(net, train_set, test_set, opt):
    # temperature setting
    T=opt.T; t=1; T_min=0.01; update_T = False; 

    writer = SummaryWriter(comment=opt.exp_name + '_' + 'fs' + '_' + opt.cooling_strategy + '_' + str(opt.T))
    best_acc = 0.0

    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=False, num_workers=2, collate_fn=MathExpr_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2, collate_fn=MathExpr_collate)

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
            seq_len = sample['len']
            img_seq = sample['img_seq']
            batch_truths = sample['label_seq']
            batch_results = sample['res']
            index = sample['index']
            expr = sample['expr']

            N, M, C, H, W = img_seq.shape
            x = img_seq.reshape(N*M, C, H, W).cuda()            
            batch_logits = net(x).reshape(N, M, -1)
            x = batch_logits.reshape(-1, num_classes)

            if epoch <= sampling_epoch: # initilization: train on uniform sampling
                batch_labels = pseudo_labels[index, :]
                pseudos, changes = walk(batch_logits, batch_labels, batch_results, 0.0)
                total_changes += changes
                pseudo_labels[index, :] = pseudos
                y = pseudos.reshape(-1).cuda()       
                cls_loss = criterion(x, y)
                optim_cls.zero_grad()
                cls_loss.backward()
                optim_cls.step()
            elif epoch > sampling_epoch and epoch % sampling_epoch == 0: # update net parameters for each sampling epoch
                batch_labels = pseudo_labels[index, :]
                y = batch_labels.reshape(-1).cuda()        
                cls_loss = criterion(x, y)
                optim_cls.zero_grad()
                cls_loss.backward()
                optim_cls.step()
                update_T = True
            else: # update y
                batch_labels = pseudo_labels[index, :]
                pseudos, changes = walk(batch_logits, batch_labels, batch_results, T)
                total_changes += changes
                pseudo_labels[index, :] = pseudos
                y = pseudos.reshape(-1).cuda()        
                cls_loss = criterion(x, y)

            train_loss += cls_loss.item()
        
            _, pred = torch.max(x, dim=-1)
            pred_acc += (pred == y).float().mean().item()

            gt_labels = batch_truths.reshape(-1)
            gt_acc += (gt_labels == y.cpu()).float().mean().item()
                
            sys.stdout.write('\r')
            ite = (batch_idx+1)
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.3f \t total_changes: %3d \t acc/gtacc %.3f/%.3f \t T %.3f'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), \
                    train_loss/ite, total_changes, pred_acc/ite, gt_acc/ite, T))
            sys.stdout.flush()   

            # eval for each batch
            _, preds = torch.max(batch_logits, -1)
            expr_preds, res_pred_all = eval_expr(preds.data.cpu().numpy(), seq_len)
            acc = equal_res(np.asarray(res_pred_all), np.asarray(batch_results)).mean()
            expr_pred_all = ''.join(expr_preds)
            expr_all = ''.join(expr)
            sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
            acc = 100*float(acc)
            sym_acc = 100*float(sym_acc)
            train_acc.append(acc)

            count += 1
            writer.add_scalars('train_accs', {'acc': acc, 'sym_acc': sym_acc}, count)

        # save best based on training set
        if np.mean(train_acc) > best_acc:
            best_acc = np.mean(train_acc)
            save(net, file_name+'_best')

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
        acc, sym_acc = evaluate(net, eval_dataloader)
        acc = 100*float(acc)
        sym_acc = 100*float(sym_acc)
        print('\n \t Test | Epoch %3d Acc@1: %2f%% Sym_acc: %.2f%%' % (epoch, acc, sym_acc))
        writer.add_scalars('val_accs', {'acc': acc, 'sym_acc': sym_acc}, epoch)
    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'lenet' + '_' + str(opt.seed) + '_' + opt.exp_name + '_' + opt.cooling_strategy
    net = models.LeNet()
    net.cuda()
    net = train(net, train_set, test_set, opt)
    save(net, file_name)
