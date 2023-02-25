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

from diagnosis import ExprTree

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch HWF Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--pretrain', default=None, type=str, help='the path of pre-trained model')
parser.add_argument('--batch_size', default=64, type=float, help='the size of min-batch')
parser.add_argument('--nstep', default=1, type=int, help='the step of back-search')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
num_epochs = 1000; 
sgd_lr = 0.1; adam_lr = 5e-4; 
reward_decay = 0.99
buffer_weight = 0.5
batch_size = opt.batch_size


train_set = MathExprDataset('train', numSamples=int(10000), randomSeed=opt.seed)
test_set = MathExprDataset('test')

def find_fix(preds, gts, seq_lens, all_probs, nstep):
    etree = ExprTree()
    best_fix_list = []
    for pred, gt, l, all_prob in zip(preds, gts, seq_lens, all_probs):
        pred = pred[:l]
        
        all_prob = all_prob[:l]
        pred_str = [id2sym(x) for x in pred]
        tokens = list(zip(pred_str, all_prob))
        etree.parse(tokens)
        fix = [-1]
        if equal_res(etree.res()[0], gt):
            fix = list(pred)
        else:
            output = etree.fix(gt, n_step=nstep)
            if output:
                fix = [sym2id(x) for x in output[0]]
        best_fix_list.append(fix)
    return best_fix_list

def train(model, train_set, test_set, opt):
    writer = SummaryWriter(comment=opt.exp_name + '_' + str(opt.nstep) + '-BS')
    
    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=False, num_workers=2, collate_fn=MathExpr_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2, collate_fn=MathExpr_collate)

    # loss 
    best_acc = 0
    elapsed_time = 0
    criterion = nn.NLLLoss(ignore_index=-1)
    softmax = torch.nn.Softmax(dim=-1) 

    reward_moving_average = None
    optim = optimizer.Adam(net.parameters(), lr=adam_lr)

    count = 0
    for epoch in range(num_epochs):
        net.train()
        for batch_idx, sample in enumerate(train_dataloader):
            img_seq = sample['img_seq']
            label_seq = sample['label_seq']
            res = sample['res']
            seq_len = sample['len']
            expr = sample['expr']

            img_seq = img_seq.cuda()
            label_seq = label_seq.cuda()
            max_len = img_seq.shape[1]
            logits = net(img_seq, mask=True)
            probs = softmax(logits)

            selected_probs, preds = torch.max(probs, -1)
            selected_probs = torch.log(selected_probs+1e-20)
            probs = torch.log(probs + 1e-20)

            rewards = compute_rewards(preds.data.cpu().numpy(), res.numpy(), seq_len)
            if reward_moving_average is None:
                reward_moving_average = np.mean(rewards)
            reward_moving_average = reward_moving_average * reward_decay \
                    + np.mean(rewards) * (1 - reward_decay)
            rewards = rewards - reward_moving_average
            
            fix_list = find_fix(preds.data.cpu().numpy(), res.numpy(), seq_len.numpy(), 
                            probs.data.cpu().numpy(), opt.nstep)
            pseudo_label_seq = []
            for fix in fix_list:
                fix = fix + [-1] * (max_len - len(fix)) # -1 is ignored index in nllloss
                pseudo_label_seq.append(fix)
            pseudo_label_seq = np.array(pseudo_label_seq)
            pseudo_label_seq = torch.tensor(pseudo_label_seq).to(device)
            loss = criterion(probs.reshape((-1, probs.shape[-1])), pseudo_label_seq.reshape((-1,)))

            optim.zero_grad()
            loss.backward()
            optim.step()
                    
            # eval for each batch
            _, preds = torch.max(probs, -1)
            expr_preds, res_pred_all = eval_expr(preds.data.cpu().numpy(), seq_len)
            acc = equal_res(np.asarray(res_pred_all), np.asarray(res)).mean()
            expr_pred_all = ''.join(expr_preds)
            expr_all = ''.join(expr)
            sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
            acc = 100*float(acc)
            sym_acc = 100*float(sym_acc)

            sys.stdout.write('\r')
            sys.stdout.write('Train | Epoch [%3d/%3d] Iter[%3d/%3d]\t\t loss: %.5f Acc@1: %.2f%% Sym_acc: %.2f%%'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), loss.item(), acc, sym_acc))
            sys.stdout.flush()   
            count += 1
            writer.add_scalars('train_accs', {'acc': acc, 'sym_acc': sym_acc}, count)

        # eval for each epoch
        acc, sym_acc = evaluate(net, eval_dataloader)
        acc = 100*float(acc)
        sym_acc = 100*float(sym_acc)
        print('\n \t Test | Epoch %3d Acc@1: %2f%% Sym_acc: %.2f%%' % (epoch, acc, sym_acc))
        writer.add_scalars('val_accs', {'acc': acc, 'sym_acc': sym_acc}, epoch)
    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'sym_net' + '_' + str(opt.seed) + '_' + opt.exp_name + '_' + str(opt.nstep) + '-BS'
    net = models.NNAOG()
    if opt.pretrain:
        net.sym_net.load_state_dict(torch.load(opt.pretrain))
    net.cuda()
    net = train(net, train_set, test_set, opt)
    save(net, file_name)
