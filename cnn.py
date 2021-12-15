import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
import random
from transformers import RobertaTokenizer, RobertaModel
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

from sklearn.metrics import f1_score, accuracy_score


videoIDs, videoSpeakers, videoLabels, videoText,\
videoAudio, videoVisual, videoSentence, trainVid,\
testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features_raw.pkl', 'rb'), encoding='latin1')


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# keys = [x for x in trainVid]

# utterance = []
# labels = []
# for vid in keys:
#     conversation = videoSentence[vid]
#     for i in range(len(conversation)):
#         text = conversation[i]
#         encoded_input = tokenizer(text, return_tensors='pt')
#         output = model(**encoded_input).pooler_output.squeeze().detach().numpy()     
#         utterance.append(output)
#         labels.append(videoLabels[vid][i])

# utterance = np.array(utterance)
# labels = np.array(labels)


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        self.cnn = nn.Sequential(
            torch.nn.Conv1d(1,32,kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32),
            torch.nn.AdaptiveMaxPool1d(600),
            
            torch.nn.Conv1d(32,64,kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            torch.nn.AdaptiveMaxPool1d(400),
            
            torch.nn.Conv1d(64,32,kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32),
            torch.nn.AdaptiveMaxPool1d(200),
            
            torch.nn.Conv1d(32,1,kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(1),
            torch.nn.AdaptiveMaxPool1d(100)
            )
        self.linear = nn.Linear(100,6)

    def forward(self, x):
        features =  self.cnn(x.unsqueeze(1)).squeeze()
        pred = self.linear(features)
        
        return features, F.log_softmax(pred,1)
    
    
# loss_weights = torch.FloatTensor([1/0.086747,
#                                       1/0.144406,
#                                       1/0.227883,
#                                       1/0.160585,
#                                       1/0.127711,
#                                       1/0.252668])
# loss_function  = nn.NLLLoss(loss_weights)

# train_X, train_y = torch.Tensor(utterance), torch.Tensor(labels)
# train_dataset = utils.TensorDataset(train_X, train_y)
# train_loader = utils.DataLoader(train_dataset, batch_size=64, shuffle=True)

# model = CNNFeatureExtractor()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# print('train...')

# for e in range(200):
#     loss_all = 0
#     preds = []
#     labels = []
#     for X, y in train_loader:
#         _, yhat = model(X)
        
#         optimizer.zero_grad()
#         loss = loss_function(yhat, y.long())
#         loss.backward()
#         optimizer.step()
        
#         loss_all += loss
        
#         preds.append(torch.argmax(yhat, 1).cpu().numpy())
#         labels.append(y.cpu().numpy())
        
#     preds  = np.concatenate(preds)
#     labels = np.concatenate(labels)
        
#     print('epoch {}, loss: {}, acc: {}'.format(e+1, loss_all.item(), round(accuracy_score(labels, preds)*100, 2)))
    
# torch.save(model, './pretrained_models/CNN')


keys = [x for x in testVid]

utterance = []
labels = []
for vid in keys:
    conversation = videoSentence[vid]
    for i in range(len(conversation)):
        text = conversation[i]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input).pooler_output.squeeze().detach().numpy()     
        utterance.append(output)
        labels.append(videoLabels[vid][i])
        
utterance = np.array(utterance)
labels = np.array(labels)

model = torch.load('./pretrained_models/CNN')

test_X, test_y = torch.Tensor(utterance), torch.Tensor(labels)
test_dataset = utils.TensorDataset(test_X, test_y)
test_loader = utils.DataLoader(test_dataset, batch_size=64, shuffle=True)

print('test...')
preds = []
labels = []
for X, y in test_loader:
    _, yhat = model(X)
    
    preds.append(torch.argmax(yhat, 1).cpu().numpy())
    labels.append(y.cpu().numpy())
    
preds  = np.concatenate(preds)
labels = np.concatenate(labels)

print('acc: {}, f1: {}'.format(round(accuracy_score(labels, preds)*100, 2), round(f1_score(labels,preds, average='weighted')*100, 2)))
