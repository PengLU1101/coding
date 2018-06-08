import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class Graph_Sent(nn.Module):
	"""
	Build a sentence graph
	Args:
		hid_sent[FloatTensor]: Batch x n_sent x d_hid
	Outs:
		sent_graph[FloatTensor]: Batch x n_sent x n_sent
	"""
	def __init__(self, d_hid):
		super(Graph_Sent, self).__init__()
		self.d_hid = d_hid
		self.fc = nn.Linear(d_hid, d_hid)

	def forward(self, hid_sent):
		batch_size, n_sent, d_hid = hid_sent.size()
		transform_sent = self.fc(hid_sent.view(batch_size*n_sent, -1))
		transform_sent = transform_sent.view(batch_size, n_sent, -1).transpose(1, 0)
		graph = torch.bmm(hid_sent, transform_sent) # batch x n_sent x n_sent
		return graph

class Graph_Attn(nn.Module):
	"""
	Graph Attention:
	Args:
		hid_sent[FloatTensor]: Batch x n_sent x d_hid
	Outs:
		sent_graph[FloatTensor]: Batch x n_sent x n_sent
	"""
	def __init__(self, graph, attn_fuc):
		super(Graph_Attn, self).__init__()
		self.graph_sent = graph
		self.attn_fuc = attn_fuc

	def forward(self, hid_sents, cur_sent):
		pass


def Importance_Score():
	

