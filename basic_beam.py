class beam(object):
    def __init__(self, beam_num, ):
        self.beam_num = beam_num
        self.hyp_idx = [["start"] for i in range(beam_num)]
        self.back_pointer = []
        self.word_idx = []
        self._score = [torch.FloatTensor([0]).unsqueeze(1)]
        self.done_flag = False

    def advance(self, out):
        """
        
        """
        n_voc = out.size(-1)
        out = out + self.score[-1].expand_as(out)
        score, idx = torch.topk(out.view(-1), beam_num, -1) # beam_num
        word_idx = idx % n_voc
        prev_idx = idx // n_voc
        word_idx = word_idx.view(beam_num, 1)
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
        hyp = torch.cat(hyp[::-1]).unsqueeze(0)
        assert hyp.dim() == 2


        return hyp # batch x seq_len