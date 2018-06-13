import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import os
import random

from Layer.RNN_Layer import *
from Layer.Embedding_Layer import *
from Attention import *
import basic_beam

teacher_forcing_ratio = .5
Max_Length_Doc = 5
Max_Length_Sent = 10
USE_CUDA = torch.cuda.is_available()


def save_model(path, model):
    model_path = os.path.join(path, 'model.pt')
    torch.save(model.state_dict(), model_path)

def read_model(path, model):
    model_path = os.path.join(path, 'model.pt')
    model.load_state_dict(torch.load(model_path))
    return model

def Cal_index(mask):
    lens_list = torch.sum(mask, dim=1).data.tolist()
    trans_idx_list = sorted(range(len(lens_list)), key=lambda x: -lens_list[x])
    re_trans_idx_list = sorted(range(len(lens_list)), key=lambda x: trans_idx_list[x])
    trans_idx = torch.LongTensor(trans_idx_list)
    re_trans_idx = torch.LongTensor(re_trans_idx_list)
    length = [lens_list[i] for i in trans_idx_list]
    if USE_CUDA:
        trans_idx = Variable(trans_idx).cuda()
        re_trans_idx = Variable(re_trans_idx).cuda()
    else:
        trans_idx = Variable(trans_idx)
        re_trans_idx = Variable(re_trans_idx)

    return trans_idx, length, re_trans_idx


class Word_Encoder(nn.Module):
    def __init__(self,  rnn):
        super(Word_Encoder, self).__init__()
        self.rnn = rnn

    def forward(self, inputs, mask):
        """
        Args:
            inputs:(FloatTensor) Batch x num_sent x seq_len x d_embedding
            mask:(FloatTensor) Batch x num_sent x seq_len
        outs:
            hid_sent:(FloatTensor) Batch x num_sent x d_hid
        """
        batch_size, n_sent, seq_len, d_em = inputs.size()
        inputs = inputs.contiguous().view(batch_size*n_sent, seq_len, -1)
        mask = mask.contiguous().view(batch_size*n_sent, -1)
        trans_idx, length, re_trans_idx = Cal_index(mask)
        inputs_indexed = torch.index_select(inputs, 0, trans_idx)
        h0 = self.rnn.init_hid(batch_size*n_sent)
        outs, hid_sent = self.rnn(inputs_indexed, h0, length) # batch*n_sent x seq_len x d_hid
        outs = torch.index_select(outs, 0, re_trans_idx)
        hid_sent = torch.index_select(hid_sent, 0, re_trans_idx).squeeze(dim=1)
        hid_sent = hid_sent.contiguous().view(batch_size, n_sent, -1) # batch x n_sent x d_hid
        return hid_sent

class Sent_Encoder(nn.Module):
    def __init__(self, rnn):
        super(Sent_Encoder, self).__init__()
        self.rnn = rnn

    def forward(self, inputs, mask):
        """
        Args:
            inputs:(FloatTensor) Batch x num_sent x d_hid
            mask:(FloatTensor) Batch x num_sent
        outs:
            hid_doc:(FloatTensor) Batch x 1 x d_hid
        """
        batch_size, n_sent, d_hid = inputs.size()
        inputs = inputs * mask.unsqueeze(-1).float()

        trans_idx, length, re_trans_idx = Cal_index(mask)
        inputs_indexed = torch.index_select(inputs, 0, trans_idx)
        h0 = self.rnn.init_hid(batch_size)
        outs, hid_doc = self.rnn(inputs_indexed, h0, length) # hid_doc: batch x d_hid
        outs = torch.index_select(outs, 0, re_trans_idx)
        hid_doc = torch.index_select(hid_doc, 0, re_trans_idx) # batch x 1 x d_hid
        return hid_doc


class Sent_Decoder(nn.Module):
    def __init__(self, rnn):
        super(Sent_Decoder, self).__init__()
        self.rnn = rnn

    def forward(self, sent_input, hid_doc):
        """
        Args:
            hid_doc: [FloatTensor] batch x 1 x d_hid
            hid_sent_dec: [FloatTensor] batch x 1 x d_hid
        Outs:
            out: [FloatTensor] batch x 1 x d_hid
            h: [FloatTensor] batch x 1 x d_hid
        """
        batch_size, _, d_hid = hid_doc.size()
        hid_sent_dec = hid_doc.transpose(1, 0)
        out, hid_sent_dec = self.rnn(sent_input, hid_sent_dec)
        return out, hid_sent_dec
    def init_hid(self, batch_size):
        hid_sent_dec = Variable(torch.zeros(batch_size, 1, self.rnn.d_hid))
        if USE_CUDA:
            hid_sent_dec = hid_sent_dec.cuda()
        return hid_sent_dec

class Word_Decoder(nn.Module):
    def __init__(self, rnn):
        super(Word_Decoder, self).__init__()
        self.rnn = rnn

    def forward(self, tgts, hid_sent):
        return self.decode(tgts, hid_sent)

    def decode(self, tgts, hid_word_dec):
        """
        Args:
            tgts:[FloatTensor] batch x 1 x d_hid
            hid_word_dec:[FloatTensor] batch x 1 x d_emb
        Outs:
            out: [FloatTensor] batch x 1 x d_hid
            h: [FloatTensor] batch x 1 x d_hid
        """
        batch_size, _, d_hid = tgts.size()
        h = hid_word_dec.transpose(1, 0)
        out, h = self.rnn(tgts, h)
        return out, h
    def init_hid(self, batch_size):
        hid_word_dec = Variable(torch.zeros(batch_size, 1, self.rnn.d_hid))
        if USE_CUDA:
            hid_word_dec = hid_word_dec.cuda()
        return hid_word_dec

class Classifier(nn.Module):
    def __init__(self, d_in, d_voc):
        super(Classifier, self).__init__()
        self.d_in = d_in
        self.d_voc = d_voc
        self.fc = nn.Linear(d_in, d_voc)
        self.logSM = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.logSM(self.fc(x))
        assert out.dim() == 3
        return out

class EncoderDecoder(nn.Module):
    def __init__(self, w_encoder, s_encoder, 
                 w_decoder, s_decoder, attn,
                 embeddings, classifier, beam_num):
        super(EncoderDecoder, self).__init__()
        self.w_encoder = w_encoder
        self.s_encoder = s_encoder
        self.w_decoder = w_decoder
        self.s_decoder = s_decoder
        self.embeddings = embeddings
        self.attn = attn
        self.classifier = classifier
        self.beam_num = beam_num
        #self.cal_loss = loss_custorm.loss_fuc(nn.NLLLoss, ignore_index=0)
        self.cal_loss = nn.NLLLoss(ignore_index=0)

    def forward(self, srcs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s):
        hid_doc, hid_sent = self.encode(srcs, src_mask_w, src_mask_s)
        return self.decode(tgt_seqs, tgt_mask_w, hid_doc, hid_sent)

    def beam_predict(self, srcs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s):
        hid_doc, hid_sent = self.encode(srcs, src_mask_w, src_mask_s)
        return self.beam_decode(tgt_seqs,
                hid_doc, hid_sent, self.beam_num)

    def encode(self, inputs, mask_w, mask_s):
        """
        args: 
            inputs: (FloatTensor) Batch x n_sent x seq_len
            mask_w: (FloatTensor) Batch x n_sent x seq_len
            mask_s: (FloatTensor) Batch x n_sent
        outs:
            hid_sent: (FloatTensor) Batch x n_sent x d_hid
            hid_doc: (FloatTensor) Batch x 1 x d_hid
        """
        hid_sent = self.w_encoder(self.embeddings(inputs), mask_w)
        hid_doc = self.s_encoder(hid_sent, mask_s)
        
        return hid_doc, hid_sent

    def decode(self, tgt_seqs, tgt_mask_w,
                enc_hid_doc, enc_hid_sent):
        batch_size, n_sent, seq_len = tgt_seqs.size()
        batch_size, n_sent_enc, d_hid = enc_hid_sent.size()
        tgt_embs = self.embeddings(tgt_seqs) # batch x n_sent x seq_len x d_emb
        total_loss = 0
        for idx_s in range(n_sent):#
            #word_state = Decode_State()
            if idx_s == 0:
                tmp = Variable(torch.LongTensor([[4]]*batch_size))
                f_prev = Variable(torch.ones(batch_size, n_sent_enc+1) * 1/(n_sent_enc+1))
                if USE_CUDA:
                    tmp = tmp.cuda()
                    f_prev = f_prev.cuda()
                dec_out_sent = self.embeddings(tmp)
                attn_sent, f_prev = self.attn(enc_hid_sent, dec_out_sent, f_prev)
                dec_hid_sent = enc_hid_doc
            dec_input = torch.cat((dec_out_sent, attn_sent), dim=-1)
            dec_out_sent, dec_hid_sent = self.s_decoder(dec_input, dec_hid_sent)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if not self.training:
                use_teacher_forcing = False
            if use_teacher_forcing:
                for idx_w in range(seq_len):
                    if idx_w == 0:
                        dec_hid_word = dec_out_sent
                        tmp = Variable(torch.LongTensor([[2]]*batch_size))
                        if USE_CUDA:
                            tmp = tmp.cuda()
                        tgts_input = self.embeddings(tmp)
    
                    dec_out_word, dec_hid_word = self.w_decoder(tgts_input, dec_hid_word)
                    out = self.classifier(dec_out_word).squeeze(1)#batch x num_voc
                    loss = self.cal_loss(out.squeeze(1), tgt_seqs[:, idx_s, idx_w])
                    total_loss += loss
                    tgts_input = tgt_embs[:, idx_s, idx_w, :].unsqueeze(1)
                dec_out_sent = self.w_encoder(tgt_embs[:, idx_s, :, :].unsqueeze(1), tgt_mask_w[:, idx_s, :].unsqueeze(1))

            else:
                mask_hold = Variable(torch.ones(batch_size, 1)).long()
                if USE_CUDA:
                    mask_hold = mask_hold.cuda()
                gen_word_list = []
                mask_list = []
                for idx_w in range(seq_len):
                    if idx_w == 0:
                        dec_hid_word = dec_hid_sent
                        tmp = Variable(torch.LongTensor([[2]]*batch_size))
                        if USE_CUDA:
                            tmp = tmp.cuda()
                        tgts_input = self.embeddings(tmp)
                    dec_out_word, dec_hid_word = self.w_decoder(tgts_input, dec_hid_word)
                    out = self.classifier(dec_out_word).squeeze(1)#batch x num_voc
                    loss = self.cal_loss(out.squeeze(1), tgt_seqs[:, idx_s, idx_w])
                    total_loss += loss
                    #score, idx = torch.max(out, dim=-1)# idx: batch x 1
                    score, idx = torch.topk(out, 1, -1)
                    generate_word = idx.detach().long()
                    #mask = torch.gt(generate_word, 0).long()
                    generate_word = generate_word * mask_hold
                    gen_word_list.append(generate_word)
                    mask_list.append(mask_hold)
                    if torch.eq(generate_word.data, 3).cpu().numpy().any():
                        mask_next = torch.ne(generate_word, 3).long()
                        mask_hold = mask_hold * mask_next

                    tgts_input = self.embeddings(generate_word)
                    #if all(generate_word.data.cpu().numpy()) == 3: # id2token[3] = "<eos>"
                    if torch.eq(generate_word, 3).cpu().data.numpy().all():
                        break

                generate_sent = self.embeddings(torch.cat(gen_word_list, dim=1))
                mask_w = torch.cat(mask_list, dim=1)

                dec_out_sent = self.w_encoder(generate_sent.unsqueeze(1), mask_w)
        return total_loss
    def beam_decode(self, tgt_seqs,
                enc_hid_doc, enc_hid_sent, beam_num):
        batch_size, n_sent, seq_len = tgt_seqs.size()
        batch_size, n_sent_enc, d_hid = enc_hid_sent.size()

        save_hyp = []
        for idx_s in range(Max_Length_Doc):#
            #word_state = Decode_State()
            if idx_s == 0:
                tmp = Variable(torch.LongTensor([[4]]*batch_size))
                f_prev = Variable(torch.ones(batch_size, n_sent_enc+1) * 1/(n_sent_enc+1))
                if USE_CUDA:
                    tmp = tmp.cuda()
                    f_prev = f_prev.cuda()
                dec_out_sent = self.embeddings(tmp)
                attn_sent, f_prev = self.attn(enc_hid_sent, dec_out_sent, f_prev)
                dec_hid_sent = enc_hid_doc
            dec_input = torch.cat((dec_out_sent, attn_sent), dim=-1)
            dec_out_sent, dec_hid_sent = self.s_decoder(dec_input, dec_hid_sent)
            beams = basic_beam.beam(beam_num, batch_size)
            for idx_w in range(Max_Length_Sent):
                if idx_w == 0:
                    dec_hid_word = dec_hid_sent
                    tmp = Variable(torch.LongTensor([2])).unsqueeze(0)
                    if USE_CUDA:
                        tmp = tmp.cuda()
                    tgts_input = self.embeddings(tmp)
                dec_out_word, dec_hid_word = self.w_decoder(tgts_input, dec_hid_word)
                out = self.classifier(dec_out_word).squeeze(1)#batch x num_voc
                generate_word, prev_idx, done_flag = beams.advance(out.detach().data) # beam_num x 1
                if done_flag:
                    break
                generate_word = Variable(generate_word).cuda() if USE_CUDA else Variable(generate_word)
                prev_idx = Variable(prev_idx).cuda() if USE_CUDA else Variable(prev_idx)
                tgts_input = self.embeddings(generate_word).unsqueeze(1)
                dec_hid_word = torch.index_select(dec_hid_word, 0, prev_idx)

            hyp = beams.get_hyp()
            save_hyp += hyp.view(-1).tolist()
            hyp = Variable(hyp).cuda() if USE_CUDA else Variable(hyp)
            generate_sent = self.embeddings(hyp)
            mask_w = torch.gt(hyp, 0).long()

            dec_out_sent = self.w_encoder(generate_sent.unsqueeze(1), mask_w)
        return save_hyp


def build_model(d_emb, d_hid, n_layers, dropout, n_voc, beam_num=5):
    print("Building model...")
    rnn_ew = GRU_Layer(d_emb, d_hid, n_layers, dropout, bidirectional=True)
    rnn_es = GRU_Layer(d_hid, d_hid, n_layers, dropout, bidirectional=True)
    rnn_dw = GRU_Layer(d_emb, d_hid, n_layers, dropout, bidirectional=False)
    rnn_ds = GRU_Layer(d_hid+d_hid, d_hid, n_layers, dropout, bidirectional=False)
    w_encoder = Word_Encoder(rnn_ew)
    s_encoder = Sent_Encoder(rnn_es)
    w_decoder = Word_Decoder(rnn_dw)
    s_decoder = Sent_Decoder(rnn_ds)
    classifier = Classifier(d_hid, n_voc)
    graphsent = Graph_Sent(d_hid, dropout)
    graphattn = Graph_Attn(graphsent)
    embeddings = Embedding_Layer(n_voc, d_emb)
    model = EncoderDecoder(w_encoder, s_encoder, 
                           w_decoder, s_decoder, graphattn,
                           embeddings, classifier, beam_num)
    if USE_CUDA:
        model = model.cuda()
    return model


def test():
    inputs = Variable(torch.eye(12, 5).view(3, 4, -1).long())
    tgts = Variable(torch.eye(12, 5).view(3, 4, -1).long())
    w_mask = torch.triu(torch.ones(5, 4)).t().unsqueeze(-1).unsqueeze(0)
    w_mask = Variable(w_mask.expand(3, -1, -1, -1))
    s_mask = torch.triu(torch.ones(4, 3)).t()
    s_mask = Variable(s_mask)

    birnn = GRU_Layer(d_input=10, d_hid=10, n_layers=1, dropout=0.5, bidirectional=False)
    birnns = copy.deepcopy(birnn)
    birnns2 = copy.deepcopy(birnn)
    birnns3 = copy.deepcopy(birnn)

    classifier = Classifier(10, 6)
    w_encoder = Word_Encoder(birnn)
    s_encoder = Sent_Encoder(birnns)
    w_decoder = Word_Decoder(birnns2)
    s_decoder = Sent_Decoder(birnns3)
    embeddings = Embedding_Layer(6, 10)
    model = EncoderDecoder(w_encoder, s_encoder, w_decoder, s_decoder, embeddings, classifier)
    if USE_CUDA:
        model = EncoderDecoder.cuda()
        inputs = inputs.cuda()
        tgts = tgts.cuda()
        w_mask = w_mask.cuda()
        s_mask = s_mask.cuda()
    hid_state = model(inputs, w_mask.long(), s_mask.long(), tgts)
    print(hid_state.size())

if __name__ == '__main__':
    test()



        


