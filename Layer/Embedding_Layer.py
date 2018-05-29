import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

class Embedding_Layer(nn.Module):
    def __init__(self, vocab, d_emb):

        super(Embedding_Layer, self).__init__()
        self.lut = nn.Embedding(vocab, d_emb)
        self.d_emb = d_emb

    def forward(self, x, norm_flag=False):
        """
        Arguments:
            x: [batch_size, (n_sent), seq_len] LongTensor
        Output:
            embeds: [batch, seq_len, d_emb] FloatTensor
        """
        if x.dim() == 3:
            batch_size, n_sent, seq_len = x.size()
            x = x.contiguous().view(batch_size*n_sent, -1)
            embeds = self.lut(x) * math.sqrt(self.d_emb) if norm_flag else self.lut(x)
            embeds = embeds.contiguous().view(batch_size, n_sent, seq_len, -1)
            assert embeds.size(3) == self.d_emb
        else:
            embeds = self.lut(x) * math.sqrt(self.d_emb) if norm_flag else self.lut(x)
        return embeds 
    def apply_weights(self, weights, fine_tune_flag=False):
        if isinstance(weights, np.array):
            self.lut.weight.data.copy_ (torch.from_numpy(weights))
        else:
            pass
            #self.lut.weight
        if not fine_tune_flag:
            for p in self.lut.parameters():
                p.requires_grad = False