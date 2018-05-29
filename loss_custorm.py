class loss_fuc:
	def __init__(self, loss, ignore_index):
		self.ignore_index = ignore_index
		self.loss = loss(ignore_index=ignore_index)
	def cal(self, preds, targets):
		preds, targets = make_match(preds, targets)
		return self.loss(preds, targets)

def make_match(preds, targets):
	if targets.dim() == 3:
		batch_size, num_sent, seq_len = targets.size()
		preds = preds.contiguous().view(batch_size*num_sent*seq_len, -1)
		targets = targets.contiguous().view(batch_size*num_sent*seq_len)
	elif targets.dim() == 2:
		batch_size, seq_len = targets.size()
		preds = preds.contiguous().view(batch_size*seq_len, -1)
		targets = targets.contiguous().view(batch_size*seq_len)
	elif targets.dim() == 1:
		batch_size = targets.size()
	return preds, targets