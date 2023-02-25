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
parser = argparse.ArgumentParser(description='PyTorch HWF Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--batch_size', default=32, type=float, help='the size of min-batch')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
parser.add_argument('--net', default='', type=str, help='load model')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

num_epochs = 100; 
adam_lr = 1e-3; 
# sgd_lr = 0.1
softmax = torch.nn.Softmax(dim=-1) 

train_set = SudoKuDataset('train') 
test_set = SudoKuDataset('test')

def train(net, train_set, test_set, opt):

    writer = SummaryWriter(comment=opt.exp_name + '_' + 'bs' + '_')

    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=True, num_workers=2)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2)

    optim = optimizer.Adam(net.parameters(), lr=adam_lr, weight_decay=5e-4)
    # optim = optimizer.SGD(net.parameters(), lr=sgd_lr)

    criterion = nn.CrossEntropyLoss()

    count = 0

    for epoch in range(num_epochs):
        size = 0
        net.train()
        for batch_idx, sample in enumerate(train_dataloader):
            img_seq = sample['input']
            res = sample['label']
            weak_res = res.clone(); weak_res[weak_res >= 5] = 0 
            index = sample['index']

            N, M1, M2, C, H, W = img_seq.shape
            x = img_seq.reshape(N*M1*M2, C, H, W).cuda()   
            logits = net(x)
            probs = softmax(logits.reshape(N, M1, M2, -1))
        
            x, y = sample_search(probs, weak_res)
            if x is not None and y is not None:
                x = x.cuda(); y = y.cuda()
                x = torch.clamp(x, min=1e-10) # avoid numerical error
                loss = criterion(x.log().reshape(-1, num_classes), y.reshape(-1))
                optim.zero_grad()
                loss.backward()
                optim.step()
                size += x.shape[0]
            else:
                size += 0
                loss = torch.Tensor([np.inf])

            # eval for each batch
            _, preds = torch.max(logits, -1)
            preds = preds.data.cpu().numpy().reshape(N, M1, M2)
            single_correct, total_correct = eval_sudoku(preds, res)
            acc = np.mean(single_correct)
            board_acc = np.mean(total_correct)
            acc = 100*float(acc)
            total_acc = 100*float(board_acc)

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t size: %3d loss: %.5f Acc@1: %.2f%% Board_acc: %.2f%%'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), size, loss.item(), acc, total_acc))
            sys.stdout.flush()   

            count += 1
            writer.add_scalars('train_accs', {'acc': acc, 'board_acc': total_acc}, count)

        # eval for each epoch
        acc, board_acc = evaluate(net, eval_dataloader)
        acc = 100*float(acc)
        board_acc = 100*float(board_acc)
        print('\n \t Test | Epoch %3d Acc@1: %2f%% Board_acc: %.2f%%' % (epoch, acc, board_acc))
        writer.add_scalars('val_accs', {'acc': acc, 'board_acc': board_acc}, epoch)
    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'lenet' + '_' + str(opt.seed) + '_' + opt.exp_name
    net = torch.load(opt.net)['net']
    net.cuda()
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2)
    sym_acc, board_acc = evaluate(net, eval_dataloader)
    sym_acc = 100*float(sym_acc)
    board_acc = 100*float(board_acc)
    print('\n \t Stage-1 model | Sym_acc: %.2f%% Board Acc@1: %2f%%' % (sym_acc, board_acc))
    net = train(net, train_set, test_set, opt)
    save(net, file_name)