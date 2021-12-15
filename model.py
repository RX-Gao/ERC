import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn.conv import MessagePassing

import numpy as np, itertools, random, copy, math
from typing import Optional, Any, Union, Callable

from torch import Tensor


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, dropout=0):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        
        kt = kx.permute(0, 2, 1)
        qkt = torch.bmm(qx, kt)
        score = torch.div(qkt, math.sqrt(self.hidden_dim))
        
        score = F.softmax(score, dim=-1)
        
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        if output.size(1) == 1:
            output = output.squeeze()
        return output, score
    

class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):

        super(MaskedEdgeAttention, self).__init__()
        
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.att = Attention(self.input_dim, n_head=1)
        self.no_cuda = no_cuda
        
    def forward(self, M, lengths, edge_ind):
        scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

        #if torch.cuda.is_available():
        if not self.no_cuda:
            scores = scores.cuda()

        for j in range(M.size(1)):
        
            ei = np.array(edge_ind[j])
            for node in range(1,lengths[j]):
                neighbour = ei[ei[:, 0] == node, 1]

                M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                _, alpha_ = self.att(M_, t)
                with torch.no_grad():
                    # print(alpha_.size())
                    scores[j, node, neighbour] = alpha_[0, 0, :].clone() 
                    # print(scores[j, node, neighbour])

        return scores


def edge_perms(l, speaker):
    perms = set()
    for i in range(1, l):
        s = []
        c = 0
        n = 0
        tau = i - 1
        while tau>=0 and c<1:
            if (speaker[tau,0] == speaker[i,0]) and (speaker[tau,0] not in s):
                perms.add((i, tau))
                c = c + 1
                s.append(speaker[tau,0])
            elif speaker[tau,0] not in s:
                perms.add((i, tau))
                n = n + 1
                s.append(speaker[tau,0])
            tau = tau - 1
            
    return list(perms)   
    
        
def batch_graphify(features, qmask, lengths, edge_type_mapping, att_model, no_cuda):
    
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        speaker = qmask[0:lengths[j], j, :]
        edge_ind.append(edge_perms(lengths[j], speaker))
        
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        speaker = qmask[0:lengths[j], j, :]
        node_features.append(features[:lengths[j], j, :])
        
        # utterance index
        perms1 = edge_perms(lengths[j], speaker)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))
    
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
        
            speaker0 = 0 if qmask[item1[0], j, 0]==1 else 1 
            speaker1 = 0 if qmask[item1[1], j, 0]==1 else 1
            
            if speaker0 == speaker1:
                edge_type.append(edge_type_mapping['0'])
            else:
                edge_type.append(edge_type_mapping['1'])
                
            # edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1)])
            
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)
    
    speaker = []
    for j in range(batch_size):
        sb = [0 if qmask[i, j, 0]==1 else 1 for i in range(lengths[j])] 
        speaker.extend(sb)
        
    speaker = torch.tensor(speaker)

    #if torch.cuda.is_available():
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()
        speaker = speaker.cuda()
    
    return speaker, node_features, edge_index, edge_norm, edge_type, edge_index_lengths 


def classify_node_features(emotions, seq_lengths, umask, linear_layer, dropout_layer, smax_fc_layer, no_cuda):
    hidden = F.relu(linear_layer(emotions))
    hidden = dropout_layer(hidden)
    hidden = smax_fc_layer(hidden)

    log_prob = F.log_softmax(hidden, 1)
    return log_prob
    

class GNN_layer(MessagePassing):

    def __init__(self, network, in_channels, out_channels, num_relations, **kwargs):
        super(GNN_layer, self).__init__(aggr='mean', flow="target_to_source", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        
        self.type = network

        self.w = nn.Parameter(torch.Tensor(num_relations, in_channels*out_channels))
            

    def forward(self, GRUs, speaker, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm, speaker=speaker, GRUs=GRUs)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = self.w
        
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_type)
        # print(w.size())
        out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x, edge_type, speaker, GRUs):
        # x:(N,in_channels)
        if self.type == 'GGNN':
            # aggr_out:(N,out_channels)
            out1 = torch.zeros(x.shape)
            for i in range(x.size(0)):
                if speaker[i]==0:
                    output, _ = GRUs[0](x[i:i+1,:].unsqueeze(0), aggr_out[i:i+1,:].unsqueeze(0))
                    out1[i,:] = output.squeeze()
                else:
                    output, _ = GRUs[1](x[i:i+1,:].unsqueeze(0), aggr_out[i:i+1,:].unsqueeze(0))
                    out1[i,:] = output.squeeze()
                    
            out2 = torch.zeros(x.shape)
            for i in range(x.size(0)):
                if speaker[i]==0:
                    output, _ = GRUs[2](aggr_out[i:i+1,:].unsqueeze(0), x[i:i+1,:].unsqueeze(0))
                    out2[i,:] = output.squeeze()
                else:
                    output, _ = GRUs[3](aggr_out[i:i+1,:].unsqueeze(0), x[i:i+1,:].unsqueeze(0))
                    out2[i,:] = output.squeeze()
                    
            out = out1 + out2
            
        else:
            out = aggr_out 

        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


class GraphNetwork(torch.nn.Module):
    def __init__(self, hidden_size, num_classes, num_relations, max_seq_len, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        
        self.gruh1 = nn.GRU(hidden_size, hidden_size, num_layers=1, bidirectional=False)
        self.gruh2 = nn.GRU(hidden_size, hidden_size, num_layers=1, bidirectional=False)
        
        self.grum1 = nn.GRU(hidden_size, hidden_size, num_layers=1, bidirectional=False)
        self.grum2 = nn.GRU(hidden_size, hidden_size, num_layers=1, bidirectional=False)
        
        
        self.gnn1 = GNN_layer('GGCN', hidden_size, hidden_size, num_relations)
        self.gnn2 = GNN_layer('GGCN', hidden_size, hidden_size, num_relations)
        
        self.linear   = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc  = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda 
        
        
    def forward(self, speaker, x, edge_index, edge_norm, edge_type, seq_lengths, umask):
        out1 = F.sigmoid(self.dropout(self.gnn1([self.gruh1, self.gruh2, self.grum1, self.grum2], speaker, x, edge_index, edge_type, edge_norm=edge_norm*2))) 
        
        out2 = (self.gnn2([self.gruh1, self.gnn2, self.grum1, self.grum2], speaker, out1, edge_index, edge_type, edge_norm=edge_norm*2))

        emotions = torch.cat((x, out2), 1)
        
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.linear, self.dropout, self.smax_fc, self.no_cuda)
        return log_prob, x, emotions
    
    
class DialogueGNNModel(nn.Module):

    def __init__(self, D_e, n_speakers, max_seq_len, n_classes=6, dropout=0.5, no_cuda=False):
        
        super(DialogueGNNModel, self).__init__()
        
        self.no_cuda = no_cuda
            
        self.TransGRU = torch.load('./pretrained_models/TransGRU')
        for param in self.TransGRU.parameters():
            param.requires_grad = False
            
        # n_relations = n_speakers ** 2
        n_relations = 2

        self.att_model = MaskedEdgeAttention(2*D_e, max_seq_len, self.no_cuda)
        self.graph_net = GraphNetwork(2*D_e, n_classes, n_relations, max_seq_len, dropout, self.no_cuda)

        edge_type_mapping = {}
        # for j in range(n_speakers):
        #     for k in range(n_speakers):
        #         edge_type_mapping[str(j) + str(k)] = len(edge_type_mapping)
        edge_type_mapping['0'] = 0
        edge_type_mapping['1'] = 1
        self.edge_type_mapping = edge_type_mapping
        

    def forward(self, U, qmask, umask, seq_lengths):
        emotions, _ = self.TransGRU(U, qmask, umask, seq_lengths)    
        
        speaker, features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, seq_lengths, self.edge_type_mapping, self.att_model, self.no_cuda)
        log_prob, embd_1, embd_2 = self.graph_net(speaker, features, edge_index, edge_norm, edge_type, seq_lengths, umask)
        
        return embd_1, embd_2, log_prob, edge_index, edge_norm, edge_type, edge_index_lengths


class DialogueGRUModel(nn.Module):

    def __init__(self, D_m, D_e, n_speakers, max_seq_len, n_classes=6,nhead=4, dropout=0.5, no_cuda=False):
        
        super(DialogueGRUModel, self).__init__()

        self.no_cuda = no_cuda
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_m, nhead=nhead)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer1 = transformer_encoder
        self.gru1 = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=2*D_e, nhead=nhead)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer2 = transformer_encoder
    
        self.linear   = nn.Linear(2*D_e, D_e)
        self.smax_fc  = nn.Linear(D_e, 6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, U, qmask, umask, seq_lengths):
        U = self.transformer1(U, src_key_padding_mask=umask)
        emotions = self.gru1(U)[0]
        emotions = self.transformer2(emotions, src_key_padding_mask=umask)
            
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.linear, self.dropout, self.smax_fc, self.no_cuda)
        
        return emotions, log_prob
    