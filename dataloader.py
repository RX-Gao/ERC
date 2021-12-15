import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import random
    
        
class IEMOCAPDataset(Dataset):
    def __init__(self, train):
        
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features_processed.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
            
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        textf, visuf, acouf, qmask, umask, label, vix = zip(*data)
        textf = pad_sequence(textf)
        visuf = pad_sequence(visuf)
        acouf = pad_sequence(acouf)
        qmask = pad_sequence(qmask)
        umask = pad_sequence(umask,True)
        label = pad_sequence(label,True)
        
        return textf, visuf, acouf, qmask, umask, label, list(vix)