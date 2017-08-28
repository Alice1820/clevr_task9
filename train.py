import pickle
from collections import Counter
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tqdm import tqdm
import time
import csv
import datetime
from dataset import CLEVR, collate_data
from model import RelationNetworks

# Training settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--n-epoch', type=int, default=1000, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--is-multi-gpu', default=False, help='if use multiple gpu(default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str, help='resume from model stored')
parser.add_argument('--data-dir', type=str, default='/home/zhangxifan/CLEVR_v1.0', help='(default: /home/zhangxifan/CLEVR_v1.0)')
parser.add_argument('--segs-dir', type=str, default='/home/zhangxifan/CLEVR_v1.0', help='(default: /home/zhangxifan/CLEVR_v1.0/images)')

#'/home/zhangxifan/CLEVR_v1.0'
#'/home/mi/RelationalReasoning/CLEVR_v1.0'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def train(epoch):
    train_set = DataLoader(CLEVR(args.data_dir, args.segs_dir, 'train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=16,
                    collate_fn=collate_data, pin_memory=args.cuda)

#    dataset = iter(train_set)
#    pbar = tqdm(dataset)
    moving_loss = 0

    relnet.train(True)
    avg_loss = 0
    avg_acc = 0
    for step, (apps, masks, num_layers, question, q_len, answer, family) in enumerate(train_set):
#        print (answer)
#        answer_value = answer
#        step += 1
        start_time = time.time()
        q_len = torch.LongTensor(np.array(q_len))
        num_layers = torch.LongTensor(np.array(num_layers))
        if args.cuda:
            apps, masks, num_layers, question, answer, q_len = \
            Variable(apps).cuda(), Variable(masks).cuda(), Variable(num_layers).cuda(),  Variable(question).cuda(), Variable(answer).cuda(), Variable(q_len).cuda()
            
        else:
            apps, masks, num_layers, question, answer, q_len = \
            Variable(apps), Variable(masks), Variable(num_layers), Variable(question), Variable(answer), Variable(q_len)

        relnet.zero_grad()
        output = relnet(apps, masks, num_layers, question, q_len)
        
        pred_answer = output.data.cpu().numpy().argmax(1)
        accuracy = np.mean(answer.data.cpu().numpy() == pred_answer)
        
        avg_acc += accuracy
        loss = torch.sum(criterion(output, answer))
        loss.backward()
        optimizer.step()
        
        if moving_loss == 0:
            moving_loss = loss.data[0]

        else:
            moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1
        
        avg_loss += loss.data[0]
#        pbar.set_description('Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'. \
#                            format(epoch + 1, loss.data[0], moving_loss))
        exm_per_sec = args.batch_size / (time.time() - start_time)
        if step % args.log_interval == 0:
            print ('{}; Epoch: {}; Step: {:d}; Loss: {:.5f}; Avg: {:.5f}; Avg_Accuracy: {:.5f}; Example/sec: {:.5f}'.format(datetime.datetime.now(), epoch, step, loss.data[0], avg_loss/(step+1), avg_acc/(step+1), exm_per_sec))
#            print ('{}; Epoch: {}; Step: {:d}; Loss: {:.5f}; Avg: {:.5f}; Avg_Accuracy: {:.5f}'.format(datetime.datetime.now(), epoch + start_from_epoch, step, loss.data[0], avg_loss/(step+1), avg_acc/(step+1)))
        
        
    with open('logs_train.csv', 'a') as csvfile_train:
        fieldnames_train = ['epoch', 'train_loss', 'train_acc']
        writer_train = csv.DictWriter(csvfile_train, fieldnames=fieldnames_train)
        writer_train.writerow({'epoch':epoch, 'train_loss':avg_loss/(step+1), 'train_acc':avg_acc/(step+1)})
            

def valid(epoch):
    valid_set = DataLoader(CLEVR(args.data_dir, args.segs_dir, 'val'),
                    batch_size=args.batch_size, shuffle=True, num_workers=16,
                    collate_fn=collate_data, pin_memory=args.cuda)
#    dataset = iter(valid_set)

    relnet.train(True)
#    step = 0
    avg_loss = 0
    avg_acc = 0
    for step, (apps, masks, num_layers, question, q_len, answer, family) in enumerate(valid_set):
#        step += 1
        q_len = torch.LongTensor(np.array(q_len))
        num_layers = torch.LongTensor(np.array(num_layers))
        if args.cuda:
            apps, masks, num_layers, question, answer, q_len = \
            Variable(apps).cuda(), Variable(masks).cuda(), Variable(num_layers).cuda(),  Variable(question).cuda(), Variable(answer).cuda(), Variable(q_len).cuda()
            
        else:
            apps, masks, num_layers, question, answer, q_len = \
            Variable(apps), Variable(masks), Variable(num_layers), Variable(question), Variable(answer), Variable(q_len)
            
        output = relnet(apps, masks, num_layers, question, q_len)
#        correct = output.data.cpu().numpy().argmax(1) == answer.data.cpu().numpy()
        pred_answer = output.data.cpu().numpy().argmax(1)
        accuracy = np.mean(answer.data.cpu().numpy() == pred_answer)
        
        avg_acc += accuracy
        loss = torch.sum(criterion(output, answer))
        avg_loss += loss.data[0]
        
        if step % args.log_interval ==0:
            print ('Epoch: {}; Step: {:d}; Loss: {:.5f}; Avg_Accuracy: {:.5f}'. \
                            format(epoch, step, loss.data[0], avg_acc/(step+1)))
    
    with open('logs_valid.csv', 'a') as csvfile_valid:
        fieldnames_valid = ['epoch', 'valid_loss', 'valid_acc']
        writer_valid = csv.DictWriter(csvfile_valid, fieldnames=fieldnames_valid)
        writer_valid.writerow({'epoch':epoch, 'valid_loss':avg_loss/(step+1), 'valid_acc':avg_acc/(step+1)})

    print('Epoch: {:d}; Avg Acc: {:.5f}'.format(epoch, avg_acc/(step+1)))


with open('data/dic.pkl', 'rb') as f:
    dic = pickle.load(f)

n_words = len(dic['word_dic']) + 1
n_answers = len(dic['answer_dic'])

#    relnet = nn.DataParallel(RelationNetworks(n_words), device_ids=[0, 1, 2, 3]).cuda()
#    relnet = nn.parallel.DataParallel(RelationNetworks(n_words)).cuda()
if args.is_multi_gpu:
    relnet = nn.parallel.DataParallel(RelationNetworks(n_words))
else:
    relnet = RelationNetworks(n_words)
    
if args.cuda:
    relnet = relnet.cuda()

if args.cuda:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.Adam(relnet.parameters(), lr=2.5e-4)

# model restore
try:
    checkpoint = torch.load('model/epoch_.pth')
    relnet.load_state_dict(checkpoint)
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")

# keep logs
csvfilename_train = 'logs_train.csv'
csvfilename_valid = 'logs_valid.csv'

if not args.resume:
    with open('logs_train.csv', 'w') as csvfile_train:
        fieldnames_train = ['epoch', 'train_loss', 'train_acc']
        writer_train = csv.DictWriter(csvfile_train, fieldnames=fieldnames_train)
        writer_train.writeheader()
    with open('logs_valid.csv', 'w') as  csvfile_valid:
        fieldnames_valid = ['epoch', 'valid_loss', 'valid_acc']
        writer_valid = csv.DictWriter(csvfile_valid, fieldnames=fieldnames_valid)
        writer_valid.writeheader()

for epoch in range(args.n_epoch):

    train(epoch)
    if epoch % 5 ==0:
        valid(epoch)
    torch.save(relnet.state_dict(), 'model/epoch_{}.pth'.format(epoch))