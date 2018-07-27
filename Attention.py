import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math
USE_CUDA = torch.cuda.is_available()


class Graph_Sent(nn.Module):
    """
    Build a sentence graph
    Args:
        hid_sent[FloatTensor]: Batch x n_sent x d_hid
    Outs:
        sent_graph[FloatTensor]: Batch x n_sent x n_sent
    """
    def __init__(self, d_hid, dropout, max_len=5000):
        super(Graph_Sent, self).__init__()
        self.d_hid = d_hid
        self.fc = nn.Linear(d_hid, d_hid)

    def forward(self, hid_sent):
        batch_size, n_sent, d_hid = hid_sent.size()
        #pos_plus_hid = self.pe(hid_sent)
        pos_plus_hid = F.normalize(hid_sent, dim=2)
        transform_sent = self.fc(pos_plus_hid.view(batch_size*n_sent, -1))
        #transform_sent = hid_sent
        transform_sent = transform_sent.view(batch_size, n_sent, -1).transpose(1, 2)
        graph_W = torch.bmm(pos_plus_hid, transform_sent) # batch x n_sent x n_sent
        col_W = torch.sum(graph_W, dim=1) # batch x n_sent
        d = []
        for vec in col_W:
            d.append(torch.diag(vec))
        graph_D = torch.stack(d)
        return graph_W, graph_D

class Graph_Attn(nn.Module):
    """
    Graph Attention:
    Args:
        hid_sents[FloatTensor]: Batch x n_sent x d_hid
        cur_sent[FloatTensor]: Batch x 1 x d_hid
        f_prev[FloatTensor]: Batch x n_sent
    Outs:
        weighted_sents[FloatTensor]: Batch x d_hid
    """
    def __init__(self, graph, lmbd=.9):
        super(Graph_Attn, self).__init__()
        self.graph_sent = graph
        self.lmbd = lmbd

    def forward(self, hid_sents, cur_sent, f_prev):
        batch_size, n_sent, d_hid = hid_sents.size()
        topic_sents = torch.cat((cur_sent, hid_sents), dim=1)
        topic_graph_W, topic_graph_D = self.graph_sent(topic_sents)
        f_curr = Importance_Score(self.lmbd, topic_graph_W, topic_graph_D)
        attn_weights = cal_attn(f_prev, f_curr)
        weighted_sents = hid_sents * attn_weights[:, 1:].unsqueeze(-1)
        score, idxes = torch.max(attn_weights, dim=1)


        return torch.sum(weighted_sents, dim=1).unsqueeze(1), f_curr, idxes # batch x d_hid
        


def Importance_Score(lmbd, W, D):
    batch_size, n_sent, n_sent = W.size()
    d = []
    for i in range(batch_size):
        d.append(torch.eye(n_sent, n_sent))
    I = Variable(torch.stack(d, dim=0))
    if USE_CUDA:
        I = I.cuda()
    inv_D = [torch.inverse(D_slice) for D_slice in torch.unbind(D, dim=0)]
    inv_D = torch.stack(inv_D, dim=0)
    score = [torch.inverse(x) for x in torch.unbind(I - lmbd*torch.bmm(W, inv_D))]
    score = torch.stack(score, dim=0)
    out = (1 - lmbd) * score[:, :, 0]
    #score = torch.inverse((I - lmbd*torch.bmm(W, torch.inverse(D))))
    return out

def themo_matrix(W, D, inv_D):
    batch_size, n_sent, n_sent = W.size() 
    d = [torch.mean(x) for x in torch.unbind(D, dim=0)]
    d = torch.stack(d, dim=0)
    P = torch.bmm(inv_D, W)
    v = D / d.unsqueeze(-1)
    u = torch.ones(batch_size, n_sent, 1)
    Q = P - torch.bmm(u, v)
    container = []
    for i in range(batch_size):
        container.append(torch.eye(n_sent, n_sent))
    I = Variable(torch.stack(container, dim=0))
    if USE_CUDA:
        I = I.cuda()
    S = I - Q
    inv_S = [torch.inverse(x) for x in torch.unbind(S, dim=0)]
    inv_S = torch.stack(inv_S, dim=0)
    return inv_S

def themo_attn(inv_S, v, u, h, idx):
    batch_size, n_sent, n_sent = inv_S.size()
    pass


def build_H(n_sent):
    H = [1] + [0] * (n_sent-1)
    H = Variable(torch.Tensor(H))
    if USE_CUDA:
        H = H.cuda()
    return H

def swap_matrix(idx, matrix):
    _ = torch.index_select(matrix, 0, idx)
    tmp = torch.index_select(_, 1, idx)
    return tmp

def swap_vector(idx, vector):
    tmp = torch.index_select(vector, 0, idx)
    return tmp

def get_idx(idx_src, idx_tgt, idx_current):
    
    
    return idx_list

def chunk_matrix(idx, matrix):
    S_11 = matrix[:idx, :idx]
    S_21 = matrix[idx:, :idx]
    return S_11, S_21

def cal_themo(S_11, S_21, h_1, v_1, v_2, u_1, u_2):
    """
    Function for calculate the themo weight.
    Args:
        S_11: [FloatTensor] 
        S_21: [FloatTensor]
        h_1: [FloatTensor]
        v_1: [FloatTensor]
        v_2: [FloatTensor]
        u_1: [FloatTensor]
        u_2: [FloatTensor]
    """
    inv_S_11 = torch.inverse(S_11)
    G_1 = S_21 @ inv_S_11 @ h_1
    G_2 = S_21 @ inv_S_11 @ u_1
    h_2 = ((v_1 @ h_1 + v_2 @ G_1) @ (u_2 - G_2)) / (1 - v_2 @ u_2 + v_2 @ G_2) + G_1
    h = torch.cat((h_1, h_2))
    _, idx = torch.max(h_2)

    return h, idx + h_1.size(1)



def cal_attn(f_prev, f_curr, smoothing=0):
    """
    Args:
        f_prev[FloatTensor]: batch x n_sent
        f_curr[FloatTensor]: batch x n_sent
    Outs:
        attn_weights[FloatTensor]: batch x n_sent
    """
    batch_size, n_sent = f_curr.size()
    zeros = Variable(torch.zeros(batch_size, n_sent) + smoothing)
    if USE_CUDA:
        zeros = zeros.cuda()
    attn_scores = torch.max(f_curr-f_prev, zeros)
    attn_sm = torch.sum(attn_scores, dim=-1).unsqueeze(-1)
    attn_weights = attn_scores / attn_sm

    return attn_weights

class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)




def test():
    hid_sents = Variable(torch.randn(3, 3, 6))
    f_prev = Variable(torch.randn(3, 4))
    cur_sent = Variable(torch.randn(3, 6))

    graphsent = Graph_Sent(6, 0.1)
    graphattn = Graph_Attn(graphsent)
    if USE_CUDA:
        graphattn = graphattn.cuda()
        hid_sents = hid_sents.cuda()
        f_prev = f_prev.cuda()
        cur_sent = cur_sent.cuda()
    w_hid_sent, f_curr = graphattn(hid_sents, cur_sent, f_prev)
    print(w_hid_sent)
    print(f_curr)

if __name__ == '__main__':
    test()