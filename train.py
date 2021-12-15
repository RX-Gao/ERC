import numpy as np
import argparse
import time
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
from model import DialogueGRUModel, DialogueGNNModel
from sklearn.metrics import f1_score, accuracy_score

seed = 42


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=False)

    return train_loader, valid_loader, test_loader


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, n_epochs, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(
        0).type(torch.LongTensor), torch.empty(0), []

    # if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for i, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [
            d.cuda() for d in data[:-1]] if cuda else data[:-1]

        lengths = [(umask[j] == 1).nonzero().tolist()[-1]
                   [0] + 1 for j in range(len(umask))]

        embd_1, embd_2, log_prob, e_i, e_n, e_t, e_l = model(
            textf, qmask, umask, lengths)
        
        for j in range(label.size(0)):
            if j == 0:
                U = textf[:lengths[j], j, :]
            else:
                U = torch.cat((U, textf[:lengths[j], j, :]), 0)
                
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        if i == 0:
            embds_1 = U
            embds_2 = embd_2
        else:
            embds_1 = torch.cat((embds_1, U), 0)
            embds_2 = torch.cat((embds_2, embd_2), 0)

        # print(log_prob)

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        embds_1 = embds_1.detach().numpy()
        embds_2 = embds_2.detach().numpy()

    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted')*100, 2)

    return embds_1, embds_2, avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el


def train_or_eval_GRU_model(model, loss_function, dataloader, epoch, n_epochs, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(
        0).type(torch.LongTensor), torch.empty(0), []

    # if torch.cuda.is_available():
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for i, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [
            d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1]
                   [0] + 1 for j in range(len(umask))]

        embd_1, log_prob = model(textf, qmask, umask, lengths)
        for j in range(label.size(0)):
            if j == 0:
                log_probs = log_prob[:lengths[j], j, :]
                embd_1_ = embd_1[:lengths[j], j, :]
            else:
                log_probs = torch.cat((log_probs, log_prob[:lengths[j], j, :]), 0)
                embd_1_ = torch.cat((embd_1_, embd_1[:lengths[j], j, :]), 0)
                
        log_prob = log_probs
        embd_1 =  embd_1_
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        
        if i == 0:
            embds_1 = embd_1
        else:
            embds_1 = torch.cat((embds_1, embd_1), 0)         

        if train:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        embds_1 = embds_1.detach().numpy()
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted')*100, 2)

    return embds_1, avg_loss, avg_accuracy, labels, preds, avg_fscore, vids


if __name__ == '__main__':

    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true',
                        default=True, help='does not use GPU')

    parser.add_argument('--lr', type=float, default=1e-4,
                        metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0,
                        metavar='L2', help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.0,
                        metavar='dropout', help='dropout rate')

    parser.add_argument('--nhead', type=int, default=10,
                        metavar='nhead', help='nhead of self attention')

    parser.add_argument('--batch-size', type=int, default=32,
                        metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=30,
                        metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true',
                        default=False, help='use class weights')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda

    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    D_m = 100
    D_e = 100

    seed_everything()
    model = DialogueGNNModel(D_e,
                              n_speakers=2,
                              max_seq_len=110,
                              n_classes=n_classes,
                              dropout=args.dropout,
                              no_cuda=args.no_cuda)
    
    model_ = torch.load('./pretrained_models/TransGRU')
    
    # model = DialogueGRUModel(D_m, D_e,
    #                           n_speakers=2,
    #                           max_seq_len=110,
    #                           n_classes=n_classes,
    #                           nhead=args.nhead,
    #                           dropout=args.dropout,
    #                           no_cuda=args.no_cuda)

    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor([1/0.086747,
                                      1/0.144406,
                                      1/0.227883,
                                      1/0.160585,
                                      1/0.127711,
                                      1/0.252668])

    if args.class_weight:
        loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = nn.NLLLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    
    max_acc = 0
    for e in range(n_epochs):
        start_time = time.time()

        _, _, train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(
            model, loss_function, train_loader, e, n_epochs, cuda, optimizer, True)
        embds_1, embds_3_, test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(
            model, loss_function, test_loader, e, n_epochs, cuda)

        # _, train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_GRU_model(model, loss_function, train_loader, e, n_epochs, cuda, optimizer, True)
        embds_2, _, _, _, _, _, _ = train_or_eval_GRU_model(model_, loss_function, test_loader, e, n_epochs, cuda)
        
        
        all_fscore.append(test_fscore)
        
        if max_acc < test_acc:
            max_acc = test_acc
            embds_3 = embds_3_
            pred = test_pred

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.
              format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))


    print('Test performance..')
    print('F-Score:', max(all_fscore))
    
    
    

