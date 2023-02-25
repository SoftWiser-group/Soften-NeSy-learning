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
num_epochs = 100; 
sgd_lr = 0.1; adam_lr = 5e-4; 
reward_decay = 0.99
buffer_weight = 0.5
batch_size = opt.batch_size


train_set = ShortPathDataset('train') 
test_set = ShortPathDataset('test')

train_paths = torch.load('./data/train_paths.pt')
test_paths = torch.load('./data/test_paths.pt')

def train(model, train_set, test_set, opt):
    writer = SummaryWriter(comment=opt.exp_name + '_' + opt.mode)

    # train/test loader
    print('train:', len(train_set), '  test:', len(test_set))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, 
                            shuffle=True, num_workers=2, collate_fn=Graph_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2, collate_fn=Graph_collate)

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
        model.train()

        if opt.mode == 'MAPO':
            counter = 0
            train_queue = []
        for batch_idx, sample in enumerate(train_dataloader):
            batch_inputs = sample['input']
            res = sample['label'] # shall not be used
            batch_indices = sample['index']
            batch_results = [train_paths[i] for i in batch_indices]

            x = batch_inputs.cuda()         
            batch_logits = net(x)
            probs = batch_logits.reshape(-1, n)

            if opt.mode == "RL":
                m = Normal(probs, torch.ones_like(probs))
                preds = m.sample()
                selected_probs = m.log_prob(preds)

                rewards = compute_rewards(preds.data.cpu(), batch_inputs,  batch_results)
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                rewards = rewards - reward_moving_average
            
                selected_probs = selected_probs.sum(dim=1)
                loss = - torch.tensor(rewards, device=device) * selected_probs
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()
                optim.step()

            elif opt.mode == "MAPO":
                 #explore
                m = Normal(probs, torch.ones_like(probs))
                preds = m.sample()
                selected_probs = m.log_prob(preds)
                
                rewards = compute_rewards(preds.data.cpu(), batch_inputs,  batch_results)
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                
                rewards = rewards - reward_moving_average
                selected_probs = selected_probs.data.cpu().numpy()
                
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
                            train_queue.append({"img_seq":batch_inputs[j],"res":batch_results[j],\
                                              "preds":preds[j],"weights":buf['probs']*buffer_weight,"rewards":reward})
                    counter += 1
                    j += 1       

                #on-policy
                m = Normal(probs, torch.ones_like(probs))
                preds = m.sample()
                selected_probs = m.log_prob(preds) 
                
                rewards = compute_rewards(preds.data.cpu(), batch_inputs,  batch_results)
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                rewards = rewards - reward_moving_average
                selected_probs = selected_probs.data.cpu().numpy()
                
                j = 0
                for reward in rewards:
                    train_queue.append({"img_seq":batch_inputs[j],"res":batch_results[j],\
                    "preds":preds[j],"weights":np.exp(selected_probs[j].sum())*(1-buffer_weight),"rewards":reward})
                    j += 1         

                batch_number = int(len(train_queue)/batch_size)
                
                loss = torch.Tensor([0.0])
                for i in range (0, batch_number-1):
                    batch_queue = train_queue[i*batch_size:(i*batch_size+batch_size)]

                    max_len = 0
                    for j in range (0, batch_size):
                        if batch_queue[j]['img_seq'].shape[0] > max_len:
                            max_len = batch_queue[j]['img_seq'].shape[0]
                            
                    img_seq = torch.zeros((batch_size,n,n),device=device)
                    preds = torch.zeros((batch_size,max_len),device=device)
                    seq_len = torch.zeros((batch_size),dtype=torch.long)
                    weights = []
                    rewards = []
                    expr = []
                    res = []
                    for j in range (0, batch_size):
                        img_seq[j] = batch_queue[j]['img_seq']
                        preds[j] = batch_queue[j]['preds']
                        weights.append(batch_queue[j]['weights'])
                        rewards.append(batch_queue[j]['rewards'])
                        res.append(batch_queue[j]['res'])
                    probs = model(img_seq)
                    m = Normal(probs, torch.ones_like(probs))
                    selected_probs = m.log_prob(preds) 
                    selected_probs = selected_probs.sum(dim=1)
                    loss = - torch.tensor(weights,device=device,dtype=torch.double) * torch.tensor(rewards, device=device,dtype=torch.double) * selected_probs
                    loss = loss.mean()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
            # eval for each batch
            # _, preds = torch.max(probs, -1)
            # expr_preds, res_pred_all = eval_expr(preds.data.cpu().numpy(), seq_len)
            # acc = equal_res(np.asarray(res_pred_all), np.asarray(res)).mean()
            # expr_pred_all = ''.join(expr_preds)
            # expr_all = ''.join(expr)
            # sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
            # acc = 100*float(acc)
            # sym_acc = 100*float(sym_acc)

            # sys.stdout.write('\r')
            # sys.stdout.write('Train | Epoch [%3d/%3d] Iter[%3d/%3d]\t\t loss: %.5f Acc@1: %.2f%% Sym_acc: %.2f%%'
            #         %(epoch, num_epochs, batch_idx+1, len(train_dataloader), loss.item(), acc, sym_acc))
            # sys.stdout.flush()   
            # count += 1
            # writer.add_scalars('train_accs', {'acc': acc, 'sym_acc': sym_acc}, count)

        # eval for each epoch
        if epoch % 100 == 0:
            acc, corr = evaluate(net, eval_dataloader)
            acc = 100*float(acc)
            print('\n \t Test | Epoch %3d Acc: %2f%% corr: %.2f' % (epoch, acc, corr))
            writer.add_scalars('val_accs', {'acc': acc, 'corr': corr}, epoch)
    writer.close()
    return net


if __name__ == "__main__":
    file_name = 'mlp' + '_' + str(opt.seed) + '_' + opt.exp_name + '_'
    module = models.VGAE()
    classifier = models.MLPNet()
    net = models.CatNet(module, classifier)
    net.cuda()
    net = train(net, train_set, test_set, opt)
    save(net, file_name)
