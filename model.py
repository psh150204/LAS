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
        h, _ = self.BLSTM(x) # batch * seq * h_dim

        # pyramid BLSTM for reducing time resolution
        for i in range(N):
            seq = h.size(1)
            h = h.view(bs, seq // 2, h_dim * 2) # batch * seq/2 * (h_dim * 2)
            h, _ = self.pBLSTM_layers(h)
        
        return h


class AttentionContext(nn.Module):
    def __init__(self):
        super(AttentionContext, self).__init__()
        self.phi = nn.Linear()
        self.psi = nn.Linear()

    def forward(self, s, h):
        # input
        # s : a tensor with size [batch, s_dim]
        # h : a tensor with size [batch, seq, h_dim]

        phi_s = self.phi(s) # [batch, dim]
        psi_h = self.psi(h) # [batch, seq, dim]
        
        e = torch.einsum('ik,ijk->ij', phi_s, psi_h) # for each batch, e = [< phi_s, psi_h_1 >, ... , < phi_s, psi_h_u >]
        alpha = F.softmax(e, dim = -1)
        c = torch.einsum('ij,ijk->ij', alpha, h) # [batch, seq, h_dim]

        return c
        

class Speller(nn.Module):
    def __init__(self):
        super(Speller, self).__init__()
        self.AttentionContext = AttentionContext()
        self.RNN = nn.LSTM(num_layers = 2)
        self.CharacterDistribution = nn.Linear()
    def forward(self, x):
        pass


