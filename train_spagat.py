from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import time
import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import glob

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)
torch.backends.cudnn.deterministic = True

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=100000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,  # 0.01 for pubmed, 0.005 for others
                    help='Initial learning rate.')
                    # 0.0085/0.01 lr 0.002 wd for citeseer -> 72.78
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=[1],
                    help='Number of hidden layers.')
parser.add_argument('--nheads', type=int, default=[8,1],
                    help='Number of attention heads.')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='elu param.')
parser.add_argument('--var_it', type=int, default=1,
                    help='Number of exp per config.')
parser.add_argument('--dropout', type=float, default=0.60,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=50,
                    help='Patience')
parser.add_argument('--model_subdir', type=str, default='exp1', 
                    help='temp model save dir')
parser.add_argument('--dataset', type=str, default='cora',  # citeseer  cora  pubmed
                    help='which dataset to use')
parser.add_argument('--logging', action='store_true', default=True,
                    help='print logging info.')
parser.add_argument('--use_bn', action='store_true', default=False,
                    help='whether use bn.')

warmup = 500

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
model_path = os.path.join('models', args.model_subdir)
if not os.path.isdir(model_path):
    os.makedirs(model_path)

matpath = os.path.join('./models_path/', args.model_subdir)
if not os.path.isdir(matpath):
    os.makedirs(matpath)

from utils import accuracy, load_data_orggcn, load_pathm, gen_pathm
from models_spagat import SpaGAT

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data_orggcn(args.dataset)

pathM = None                # initial pathM to None
if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    

def train(epoch, logging=args.logging, mode='GAT'):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, pathM=pathM, mode=mode)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if logging and epoch%1==0:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    return acc_val.item(), loss_val.item() 


def test(idx=idx_test, logging=False, genPath=False, mode=''):
    model.eval()
    output = model(features, adj, pathM=pathM, genPath=genPath, mode=mode)
    loss_test = F.nll_loss(output[idx], labels[idx])
    acc_test = accuracy(output[idx], labels[idx])
    if logging and args.logging:
        print("{:s} Test set results:".format(mode),
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item(), loss_test.item()


Nratio = 1.0
Ndeg = 0.5

##### one model test #####
for var in range(args.var_it):
    model = SpaGAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nheads=args.nheads,
            alpha=args.alpha
            )
    # optimizer
    optimizer = optim.Adam(model.parameters(),
                    lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
    # Train model
    t_total = time.time()
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    mode = 'GAT'
    
    startEpoch = 0
    for epoch in range(args.epochs):
        loss_value = train(epoch, mode=mode)[1]
        
        loss_value = test(idx_val, mode=mode)[1]
        torch.save(model.state_dict(), model_path+'/{}.pkl'.format(epoch))
        if loss_value < best:
            best = loss_value
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
    
        if bad_counter >= args.patience and epoch>startEpoch+warmup:
            files = glob.glob(model_path+'/*.pkl')
            for file in files:
                filebase = os.path.basename(file)
                epoch_nb = int(filebase.split('.')[0])
                if epoch_nb > best_epoch:
                    os.remove(file)
            
            model.load_state_dict(torch.load(model_path+'/{}.pkl'.format(best_epoch)))

            if mode == 'GAT':
                genPath = matpath
                test(genPath=genPath, logging=True, mode=mode)
                gen_pathm([8], matpath=matpath, Nratio=Nratio, Ndeg=Ndeg)
                pathM = load_pathm(args.dataset, matpath=matpath)
                mode = 'SPAGAN'
            elif mode == 'SPAGAN':
                test(logging=True, mode=mode)
                break
            startEpoch = epoch
            bad_counter = 0
            best = args.epochs + 1
            best_epoch = 0

        files = glob.glob(model_path+'/*.pkl')
        for file in files:
            filebase = os.path.basename(file)
            epoch_nb = int(filebase.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
        
    files = glob.glob(model_path+'/*.pkl')
    for file in files:
        filebase = os.path.basename(file)
        epoch_nb = int(filebase.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    model.load_state_dict(torch.load(model_path+'/{}.pkl'.format(best_epoch)))

    # Testing
    print('best result at epoch: {:d}'.format(best_epoch))
