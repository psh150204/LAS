import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Listener(nn.Module):
    def __init__(self, x_dim, h_dim, N):
        super(Listener, self).__init__()
        self.BLSTM = nn.LSTM(x_dim, h_dim, bidirectional = True, batch_first = True)
        self.pBLSTM_layers = get_clones(nn.LSTM(h_dim * 2, h_dim, bidirectional = True, batch_first = True), N)

    def forward(self, x):
        # input : a tensor with size [batch, seq, x_dim]
        # Assumption : seq % (2 ** N) == 0
        bs = x.size(0)

        # bottom bidirectional LSTM
        h, _ = self.BLSTM(x) # [batch, seq, h_dim]

        # pyramid BLSTM for reducing time resolution
        for i in range(N):
            seq = h.size(1)
            h = h.view(bs, seq // 2, h_dim * 2) # [batch, seq/(2**(i+1)), (h_dim * 2)]
            h, _ = self.pBLSTM_layers(h)
        
        return h # [batch, seq/(2**N), h_dim]

class OneHotVectorEncoding(nn.Module):
    def __init__(self, vocab_size):
        super(OneHotVectorEncoding, self).__init__()
        self.dim = vocab_size

    def forward(self, x):
        # input : a tensor with size [batch, seq]
        batch_size = x.size(0)
        sentence_length = x.size(1)

        # embedding
        one_hot_encoding = Variable(torch.zeros(batch_size, sentence_length, self.dim))
        #one_hot_encoding = Variable(torch.zeros(batch_size, sentence_length, self.dim)).cuda()
        for i in range(batch_size):
            for j in range(sentence_length):
                one_hot_encoding[i][j][x[i][j]] = 1

        return one_hot_encoding # [batch, seq, vocab_size]

class AttentionContext(nn.Module):
    def __init__(self, s_dim, h_dim, dim):
        super(AttentionContext, self).__init__()
        self.phi = nn.Linear(s_dim, dim)
        self.psi = nn.Linear(h_dim, dim)

    def forward(self, s, h):
        # input
        # s : a tensor with size [batch, s_dim]
        # h : a tensor with size [batch, seq, h_dim]

        phi_s = self.phi(s) # [batch, dim]
        psi_h = self.psi(h) # [batch, seq, dim]
        
        e = torch.einsum('ik,ijk->ij', phi_s, psi_h) # for each batch, e = [< phi_s, psi_h_1 >, ... , < phi_s, psi_h_u >]
        alpha = F.softmax(e, dim = -1)
        c = torch.einsum('ij,ijk->ij', alpha, h) # [batch, seq, h_dim]

        return c # [batch, seq, h_dim]
        

class Speller(nn.Module):
    def __init__(self, s_dim, h_dim, dim, embedding, sampling_rate = 0.1):
        super(Speller, self).__init__()
        self.AttentionContext = AttentionContext(s_dim, h_dim, dim)
        self.first_RNN = nn.LSTMCell()
        self.second_RNN = nn.LSTM()
        self.CharacterDistribution = nn.Linear()
        self.sampling_rate = sampling_rate
        self.embedding = embedding

    def decode(self, probs):
        

    def forward(self, Y, h, ground_truth = None):
        # input
        # Y = batch * [<sos>, y1, ..., y_(s-1)]
        # ground_truth = batch * [<sos>, y1, ..., y_n, <eos>]
        # output : Y' = batch * [y1, ..., y_s] / y_s can be <eos>
        
        s = torch.zeros() # hidden state
        cs = torch.zeros() # cell state
        probs = []

        for i in range(len(y)):
            isTruth = False

            # training
            if ground_truth != None and np.random.uniform(0,1,1)[0] > 0.1 :
                isTruth = True

            if isTruth:
                y = self.embedding(ground_truth[:, i])
            else:
                y = self.embedding(Y[:, i]) # [batch, vocab_size]
            
            s, cs = self.first_RNN(y, (s,cs))
            s = self.AttentionContext(s, h)
            s, _ = self.second_RNN(s)
            p = F.softmax(self.CharacterDistribution(s), dim = -1)
            probs.append(p)
        
        return decode(probs)

























