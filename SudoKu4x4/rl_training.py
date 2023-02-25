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

# Dataloader
parser = argparse.ArgumentParser(description='PyTorch HWF Logic Training')
parser.add_argument('--seed', default=1, type=int, help='Random seed to use.')
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--pretrain', default=None, type=str, help='the path of pre-trained model')
parser.add_argument('--batch_size', default=64, type=float, help='the size of min-batch')
parser.add_argument('--mode', default='RL', type=str, choices=['RL', 'MAPO'], help='the training strategy')
parser.add_argument('--exp_name', default='', type=str, help='Experiment name')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
num_epochs = 1000; 
sgd_lr = 0.1; adam_lr = 5e-4; 
reward_decay = 0.99
buffer_weight = 0.5
batch_size = opt.batch_size
num_classes = 4


train_set = SudoKuDataset('train') 
test_set = SudoKuDataset('test')


def train(model, train_set, test_set, opt):
    writer = SummaryWriter(comment=opt.exp_name + '_' + opt.mode)

    # train/test loader
    print('\n train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=False, num_workers=2)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2)

    # loss 
    best_acc = 0
    elapsed_time = 0
    criterion = nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=-1) 

    reward_moving_average = None
    optim = optimizer.Adam(net.parameters(), lr=adam_lr)


    if opt.mode == "MAPO":
        buffer = [[] for _ in range(len(train_set))]

    count = 0
    for epoch in range(num_epochs):
        train_acc = []
        model.train()

        if opt.mode == 'MAPO':
            counter = 0
            train_queue = []
        for batch_idx, sample in enumerate(train_dataloader):
            img_seq = sample['input']
            batch_truths = sample['label'] # shall not be used
            index = sample['index']

            N, M1, M2, C, H, W = img_seq.shape
            x = img_seq.reshape(N*M1*M2, C, H, W).cuda()            
            logits = net(x).reshape(N, M1, M2, -1)
            probs = softmax(logits)

            if opt.mode == "RL":
                m = Categorical(probs = probs)
                preds = m.sample()
                selected_probs = m.log_prob(preds)

                rewards = compute_rewards(preds.data.cpu().numpy())
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                rewards = rewards - reward_moving_average
            
                selected_probs = selected_probs.double().reshape(N, -1)
                selected_probs = selected_probs.sum(dim=1)
                loss = - torch.tensor(rewards, device=device) * selected_probs
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()
                optim.step()

            elif opt.mode == "MAPO":
                 #explore
                m = Categorical(probs = probs)
                preds = m.sample()
                selected_probs = m.log_prob(preds)
                
                rewards = compute_rewards(preds.data.cpu().numpy())
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                
                rewards = rewards - reward_moving_average
                selected_probs = selected_probs.data.cpu().numpy().reshape(N, -1)
                
                j = 0
                for reward in rewards:
                    if reward > 0:
                        flag = 0
                        for buf in buffer[counter]:
                            if buf['preds'] == preds[j].data.tolist():
                                buf['probs'] = np.exp(selected_probs[j].sum())
                                flag = 1
                        if not flag:
                            buffer[counter].append({"preds":preds[j].data.tolist(),"probs":np.exp(selected_probs[j].sum())})
                            
                        total_probs = 0
                        
                        #Re-calculate the weights in buffer
                        for buf in buffer[counter]:
                            total_probs += buf['probs']
                        for buf in buffer[counter]:
                            buf['probs'] = buf['probs']/total_probs 
                            train_queue.append({"img_seq":img_seq[j],"preds":preds[j],"weights":buf['probs']*buffer_weight,"rewards":reward})
                    counter += 1
                    j += 1       

                #on-policy
                m = Categorical(probs = probs)
                preds = m.sample()
                selected_probs = m.log_prob(preds) 
                
                rewards = compute_rewards(preds.data.cpu().numpy())
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                rewards = rewards - reward_moving_average
                selected_probs = selected_probs.data.cpu().numpy().reshape(N, -1)
                
                j = 0
                for reward in rewards:
                    train_queue.append({"img_seq":img_seq[j],"preds":preds[j],"weights":np.exp(selected_probs[j].sum())*(1-buffer_weight),"rewards":reward})
                    j += 1         

                batch_number = int(len(train_queue)/batch_size)
                
                loss = torch.Tensor([0.0])
                for i in range (0, batch_number-1):
                    batch_queue = train_queue[i*batch_size:(i*batch_size+batch_size)]

                    max_len = 0
                    for j in range (0, batch_size):
                        if batch_queue[j]['img_seq'].shape[0] > max_len:
                            max_len = batch_queue[j]['img_seq'].shape[0]
                            
                    img_seq = torch.zeros((batch_size,4,4,1,32,32),device=device)
                    preds = torch.zeros((batch_size,4,4),device=device)
                    weights = []
                    rewards = []
                    expr = []
                    res = []
                    for j in range (0, batch_size):
                        img_seq[j] = batch_queue[j]['img_seq']
                        preds[j] = batch_queue[j]['preds']
                        weights.append(batch_queue[j]['weights'])
                        rewards.append(batch_queue[j]['rewards'])
                    N_, M1, M2, C, H, W = img_seq.shape
                    img_seq = img_seq.reshape(N_*M1*M2, C, H, W).cuda()            
                    probs = softmax(model(img_seq)).reshape(N_, M1, M2, -1)
                    m = Categorical(probs = probs)
                    selected_probs = m.log_prob(preds) 
                    selected_probs = selected_probs.double().reshape(N_, -1)
                    selected_probs = selected_probs.sum(dim=1)
                    loss = - torch.tensor(weights,device=device,dtype=torch.double) * torch.tensor(rewards, device=device,dtype=torch.double) * selected_probs
                    loss = loss.mean()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
            # eval for each batch
            _, preds = torch.max(logits, -1)
            preds = preds.data.cpu().numpy().reshape(N, M1, M2)
            single_correct, total_correct = eval_sudoku(preds, batch_truths)
            acc = np.mean(single_correct)
            board_acc = np.mean(total_correct)
            acc = 100*float(acc)
            total_acc = 100*float(board_acc)
            train_acc.append(board_acc)

            sys.stdout.write('\r')
            sys.stdout.write('Train | Epoch [%3d/%3d] Iter[%3d/%3d]\t\t loss: %.5f Acc@1: %.2f%% Board_acc: %.2f%%'
                    %(epoch, num_epochs, batch_idx+1, len(train_dataloader), loss.item(), acc, board_acc))
            sys.stdout.flush()   

            count += 1
            writer.add_scalars('train_accs', {'acc': acc, 'board_acc': total_acc}, count)

        # save best based on training set
        if np.mean(train_acc) > best_acc:
            best_acc = np.mean(train_acc)
            save(net, file_name+'_best')
            
        # eval for each epoch
        acc, board_acc = evaluate(net, eval_dataloader)
        acc = 100*float(acc)
        board_acc = 100*float(board_acc)
        print('\n \t Test | Epoch %3d Acc@1: %2f%% Board_acc: %.2f%%' % (epoch, acc, board_acc))
        writer.add_scalars('val_accs', {'acc': acc, 'board_acc': board_acc}, epoch)
    return net


if __name__ == "__main__":
    file_name = 'mlp' + '_' + str(opt.seed) + '_' + opt.exp_name + '_' + opt.mode
    net = models.LeNet(num_classes)
    net.cuda()
    net = train(net, train_set, test_set, opt)
    save(net, file_name)
