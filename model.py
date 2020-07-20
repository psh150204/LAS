import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Listener(nn.Module):
    def __init__(self, x_dim, h_dim, N):
        super(Listener, self).__init__()
        self.N = N
        self.h_dim = h_dim
        self.BLSTM = nn.LSTM(x_dim, h_dim, bidirectional = True, batch_first = True)
        self.pBLSTM_layers = get_clones(nn.LSTM(h_dim * 4, h_dim, bidirectional = True, batch_first = True), N)

    def forward(self, x):
        # input : a tensor with size [batch, seq, x_dim]
        # Assumption : seq % (2 ** N) == 0
        bs = x.size(0)

        # bottom bidirectional LSTM : output dimension is double of h_dim
        h, _ = self.BLSTM(x) # [batch, seq, 2 * h_dim]
        
        # pyramid BLSTM for reducing time resolution
        for i in range(self.N):
            seq = h.size(1)
            h = h.contiguous().view(bs, seq // 2, -1) # [batch, seq/(2**(i+1)), (h_dim * 4)]
            h, _ = self.pBLSTM_layers[i](h)
        
        return h # [batch, seq/(2**N), 2 * h_dim]

class OneHotVectorEncoding(nn.Module):
    def __init__(self, num_class, device):
        super(OneHotVectorEncoding, self).__init__()
        self.dim = num_class
        self.device = device

    def forward(self, x):
        # input : a tensor with size [batch, seq]
        batch_size = x.size(0)
        sentence_length = x.size(1)

        # embedding
        one_hot_encoding = Variable(torch.zeros(batch_size, sentence_length, self.dim)).to(self.device)
        for i in range(batch_size):
            for j in range(sentence_length):
                one_hot_encoding[i][j][x[i][j]] = 1

        return one_hot_encoding # [batch, seq, num_class]

class AttentionContext(nn.Module):
    def __init__(self, s_dim, h_dim, dim):
        super(AttentionContext, self).__init__()
        self.phi = nn.Linear(s_dim, dim)
        #self.phi = nn.Sequential(
        #            nn.Linear(s_dim, dim),
        #            nn.ReLU())
        self.psi = nn.Linear(h_dim, dim)
        #self.psi = nn.Sequential(
        #            nn.Linear(h_dim, dim),
        #            nn.ReLU())

    def forward(self, s, h):
        # input
        # s : a tensor with size [batch, s_dim]
        # h : a tensor with size [batch, seq, h_dim]

        phi_s = self.phi(s) # [batch, dim]
        psi_h = self.psi(h) # [batch, seq, dim]
        
        e = torch.bmm(phi_s, psi_h.transpose(-2, -1)).squeeze(1) # [batch, seq]
        alpha = F.softmax(e, dim = -1) # [batch, seq]
        c = torch.sum(h*alpha.unsqueeze(2).repeat(1,1,h.size(2)), dim = 1)
        
        return c # [batch, h_dim]
        
# No LM rescoring / teacher forcing
class Speller(nn.Module):
    def __init__(self, s_dim, h_dim, attn_dim, num_class, device):
        super(Speller, self).__init__()
        self.AttentionContext = AttentionContext(s_dim, 2 * h_dim, attn_dim)
        self.RNN = nn.LSTM(num_class + 2 * h_dim, s_dim, num_layers=2, batch_first = True)
        self.CharacterDistribution = nn.Linear(s_dim + 2 * h_dim, num_class)
        self.embedding = OneHotVectorEncoding(num_class, device)

    def forward(self, x, h, tf_rate = 0.5):
        # input
        # x : [batch, n] = batch * [y1, ..., y_n] where y_1 is a character
        # h : [batch, seq, 2 * h_dim]
        
        bs = h.size(0)
        n = x.size(1)
        
        rnn_input = self.embedding(x[:,0:1]) # [batch, 1, num_class]
        rnn_input = torch.cat([rnn_input, h[:,0:1,:]], dim = -1) # [batch, 1, num_class + 2 * h_dim]
        
        results = None
        states = None
        
        for i in range(n):
            s, states = self.RNN(rnn_input, states) # [batch, 1, s_dim]
            c = self.AttentionContext(s, h) # [batch, 2 * h_dim]
            p = self.CharacterDistribution(torch.cat([s.squeeze(1), c], dim = -1)) # [batch, num_class]            
            
            if results is None:
                results = p.unsqueeze(1)
            else:
                results = torch.cat([results, p.unsqueeze(1)], dim = 1)
            
            if i+1 < n :
                # teacher forcing
                if np.random.uniform() < tf_rate :
                    next_char = self.embedding(x[:,i+1:i+2]) # [batch, 1, num_class]
                else :
                    pred = torch.argmax(F.softmax(p, dim = -1), dim = -1).unsqueeze(1) # [batch, 1]
                    next_char = self.embedding(pred)
                
                rnn_input = torch.cat([next_char, c.unsqueeze(1)], dim = -1) # [batch, 1, num_class + 2 * h_dim]
                
        return results
