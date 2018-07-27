import torch
from torch.autograd import Variable
USE_CUDA = torch.cuda.is_available()

class beam(object):
    def __init__(self, beam_num, batch_size):
        self.beam_num = beam_num
        self.hyp_idx = [["start"] for i in range(beam_num)]
        self.back_pointer = []
        self.word_idx = []
        score = torch.FloatTensor([[0]*batch_size])
        if USE_CUDA:
            score = score.cuda()
        self._score = [score]
        self.done_flag = False

    def advance(self, out):
        """
        
        """
        beam_num = self.beam_num
        n_voc = out.size(-1)
        out = out + self._score[-1].expand_as(out)
        score, idx = torch.topk(out.view(-1), beam_num, -1) # beam_num
        word_idx = torch.from_numpy(idx.cpu().numpy() % n_voc).cuda() if USE_CUDA else torch.from_numpy(idx.cpu().numpy() % n_voc)
        prev_idx = torch.from_numpy(idx.cpu().numpy() // n_voc).cuda() if USE_CUDA else torch.from_numpy(idx.cpu().numpy() % n_voc)
        score = score.view(beam_num, 1)
        self._score.append(score)
        self.back_pointer.append(prev_idx)
        self.word_idx.append(word_idx)
        if all(word_idx.cpu().numpy() == 3):
            self.done_flag = True
        #idx = idx.view(beam_num, 1)
        return word_idx, prev_idx, self.done_flag


    def get_hyp(self, ):
        hyp = []
        pointer_curr = 0
        for word, pointer_prev in zip(self.word_idx[::-1], self.back_pointer[::-1]):
            hyp.append(word[pointer_curr])
            pointer_curr = pointer_prev[pointer_curr]
        hyp = torch.LongTensor([hyp[::-1]])
        assert hyp.dim() == 2

        return hyp # batch x seq_len
    def cal_bigram(self, ref):
        pass

