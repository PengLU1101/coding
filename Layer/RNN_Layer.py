import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


USE_CUDA = torch.cuda.is_available()

class BiLSTM_Layer(nn.Module):
    def __init__(self, d_input, d_hid, n_layers, dropout):
        super(BiLSTM_Layer, self).__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.dropout_p = dropout

        assert d_hid % 2 == 0
        self.n_direction = 2
        self.d_hid = d_hid // 2
        self.rnn = nn.LSTM(d_input, self.d_hid, n_layers, dropout, bidirectional=True)
    def forward(self, in_seqs, in_lens):
        """
        Arguments:
            in_seqs: [batch_size, seq_len, d_input] FloatTensor
            in_lens: [batch_size] (0 - seq_len) list
        Output:
            outs: [batch, seq_len, d_hid] FloatTensor
        """
        batch_size, seq_len, d_input = in_seqs.size()
        assert d_input == self.d_input

        packed_inputs = pack(in_seqs.transpose(1, 0), in_lens)
        h0, c0 = self.init_hid(batch_size)
        outs, (h, c) = self.rnn(packed_inputs, (h0, c0))
        outs, _ = unpack(outs, in_lens)
        assert outs.size(0) == batch_size
        return outs, h.transpose(1, 0)
    def init_hid(self, batch_size):
        h = Variable(torch.zeros(self.n_layers*self.n_direction, batch_size, self.d_hid))
        c = Variable(torch.zeros(self.n_layers*self.n_direction, batch_size, self.d_hid))
        if USE_CUDA:
            h, c = h.cuda(), c.cuda()
        return h, c

class GRU_Layer(nn.Module):
    def __init__(self, d_input, d_hid, n_layers, dropout, bidirectional=False):
        super(GRU_Layer, self).__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.dropout_p = dropout
        if bidirectional:
            self.n_direction = 2
            assert d_hid % 2 == 0
            self.d_hid = d_hid // 2
        else:
            self.n_direction = 1
            self.d_hid = d_hid
        self.rnn = nn.GRU(d_input, self.d_hid, n_layers, dropout, bidirectional=bidirectional)
    def forward(self, in_seqs, h0, in_lens=None):
        """
        Arguments:
            in_seqs: [batch_size, seq_len, d_input] FloatTensor
            in_lens: [batch_size] (0 - seq_len) list
        Output:
            outs: [batch, seq_len, d_hid] FloatTensor
        """
        batch_size, seq_len, d_input = in_seqs.size()
        assert d_input == self.d_input
        if in_lens is not None:
            packed_inputs = pack(in_seqs.transpose(1, 0), in_lens)
            outs, h = self.rnn(packed_inputs, h0)
            outs, _ = unpack(outs, in_lens)
            assert outs.size(0) == batch_size
        else:
            outs, h = self.rnn(in_seqs.transpose(1, 0), h0)
            outs = outs.transpose(1, 0)
        if self.n_direction == 2:
            h_final = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
        else:
            h_final = h
        return outs, h_final.transpose(1, 0)
    def init_hid(self, batch_size):
        h = Variable(torch.zeros(self.n_layers*self.n_direction, batch_size, self.d_hid))
        if USE_CUDA:
            h = h.cuda()
        return h
