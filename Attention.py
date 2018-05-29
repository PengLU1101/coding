class Attention(nn.Module):
	def __init__(self, ):
		super(Attention, self).__init__()
		pass

	def forward(self, key, value, query):
		pass

def dot_attn(key, value, query):
	pass

class bilinear_attn(nn.Module):
	def __init__(self, ):
		super(bilinear_attn, self).__init__()

		pass
	def forward(self, key, value, query):
		pass


class mlp_attn(nn.Module):
	def __init__(self, ):
		super(mlp_attn, self).__init__()

		pass

	def forward(self, key, value, query):
		pass
	
def build_sent_graph(nn.Module):
	"""
	Function for building the original sentences graph
	args:
		sents[FloatTensor]: batch_size x n_sent x d_hid
	outs:
		graph_matric[FloatTensor]: batch x n_sent x n_sent
	"""
	def __init__(self, d_hid):
		super(build_sent_graph).__init__()
		self.d_hid = d_hid
		self.fc = nn.Linear(d_hid, d_hid)

	def forward(self, sents):
		batch_size, n_sent, d_hid = sents.size()
		sents = sents.contiguours().view(batch_size*n_sent, -1)
		transform_sents = self.fc(sents)
		transform_sents = transform_sents.contiguous()view(batch_size, n_sent, -1)
		graph_matrix = torch.bmm(sents, transform_sents.transpose(1, 2))
	return graph_matrix



