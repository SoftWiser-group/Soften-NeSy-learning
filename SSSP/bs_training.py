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
from smt_solver import sample_search, check, init_check

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch SSSP Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--batch_size', default=64, type=float, help='the size of min-batch')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--net', default='', type=str, help='load model')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

num_epochs = 30; 
adam_lr = 1e-3; 
# sgd_lr = 0.1
softmax = torch.nn.Softmax(dim=-1) 

train_set = ShortPathDataset('train') 
test_set = ShortPathDataset('test')

# save gt paths
train_graphs = torch.load('./data/train_graphs.pt')
test_graphs = torch.load('./data/test_graphs.pt')
train_paths = torch.load('./data/train_paths.pt')
test_paths = torch.load('./data/test_paths.pt')

pseudo_labels = torch.load('./data/pseudo_labels.pt')

def train(net, train_set, test_set, opt):

    writer = SummaryWriter(comment=opt.exp_name + '_' + 'bs' + '_')

    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=True, num_workers=2, collate_fn=Graph_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2, collate_fn=Graph_collate)

    optim = optimizer.Adam(net.parameters(), lr=adam_lr, weight_decay=1e-3)
    # optim = optimizer.SGD(net.parameters(), lr=sgd_lr)

    criterion = nn.MSELoss()
    count = 0

    for epoch in range(num_epochs):
        size = 0
        net.train()
        for batch_idx, sample in enumerate(train_dataloader):
            batch_inputs = sample['input']
            res = sample['label'] # shall not be used
            batch_indices = sample['index']
            batch_results = [train_paths[i] for i in batch_indices]

            x = batch_inputs.cuda()         
            batch_logits = net(x)
            x = batch_logits.reshape(-1, n)
            y = pseudo_labels[batch_indices, :]
        
            # x, y = sample_search(preds, batch_indices)
            if x is not None and y is not None:
                x = x.cuda(); y = y.cuda()
                loss = criterion(x.reshape(-1), y.reshape(-1))
                optim.zero_grad()
                loss.backward()
                optim.step()
                size += x.shape[0]
            else:
                size += 0
                loss = torch.Tensor([np.inf])

            # eval for each batch
            preds = batch_logits.clone()
            path_preds, dist_pred = eval_path(batch_inputs, preds.data.cpu())
            correct, count = equal_res(path_preds, batch_results)
            acc = 100*float(correct/count)
            corrs = []
            for p, r in zip(dist_pred, res.data.cpu().numpy()):
                corr, pval = stats.spearmanr(p, r)
                corrs.append(corr)
            corr = np.array(corrs).mean()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t size: %3d loss: %.5f Acc@1: %.2f%% Corr: %.2f'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), size, loss.item(), acc, corr))
            sys.stdout.flush()   

            count += 1
            writer.add_scalars('train_accs', {'acc': acc, 'corr': corr}, count)

        # eval for each epoch
        acc, corr = evaluate(net, eval_dataloader)
        acc = 100*float(acc)
        print('\n \t Test | Epoch %3d Acc: %2f%% corr: %.2f' % (epoch, acc, corr))
        writer.add_scalars('val_accs', {'acc': acc, 'corr': corr}, epoch)
    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'mlp' + '_' + str(opt.seed) + '_' + opt.exp_name
    net = torch.load(opt.net)['net']
    net.cuda()
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2, collate_fn=Graph_collate)
    acc, corr = evaluate(net, eval_dataloader)
    acc = 100*float(acc)
    print('\n \t Stage-1 model | Acc: %.2f%% Corr: %2f%%' % (acc, corr))
    # net = train(net, train_set, test_set, opt)
    # save(net, file_name)