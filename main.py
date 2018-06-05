import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import os
import random
import time
import math
import numpy as np
import argparse
import pickle
from tqdm import tqdm

import data_loader
import Model
import optim_custorm
import loss_custorm
from argsuse import *
import preprocess
from logger import Logger

USE_CUDA = torch.cuda.is_available()

n_voc = len(data_loader.Dataset(args.pkl_path+"train.pkl").token2id)
train_loader = data_loader.get_loader(args.pkl_path+"train.pkl", True, args.batch_size)
val_loader = data_loader.get_loader(args.pkl_path+"val.pkl", False, 1)
test_loader = data_loader.get_loader(args.pkl_path+"test.pkl", False, 1)
weight = preprocess.read_pkl(args.pkl_path+"embeddings.pkl")




def main():
    critorion = loss_custorm.loss_fuc(nn.NLLLoss, ignore_index=0)
    model = Model.build_model(args.d_emb, args.d_hid, args.n_layers, args.dropout, n_voc, args.beam_num)
    logger = Logger('./logs')

    params = model.parameters()
    model_optim = optim_custorm.NoamOpt(args.d_hid, args.factor, args.warm, torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.L2))
    model.embeddings.apply_weights(weight)
    print(model)
    if args.mode == "train":
        print("Begin training...")
        start_time = time.time()
        best_loss = 0
        for epoch_idx in range(args.max_epoch):
            val_loss, train_loss = run_epoch(model, critorion, model_optim, epoch_idx)
            print('-' * 70)
            print('| val_loss: %4.4f | train_loss: %4.4f' %(val_loss, train_loss))
            print('-' * 70)
            if not best_loss or best_loss > val_loss:
                Model.save_model(args.model_path+args.gpu+"/", model)

            if epoch_idx % 5 == 0 or not epoch_idx:
                predict(model)
    else:
        model = Model.read_model(args.model_path, model)
        save_hyp = pridict(model)


def wrap_variable(*args):
    return [Variable(tensor, requires_grad=False).cuda() if USE_CUDA else Variable(tensor, requires_grad=False) for tensor in args]

def run_epoch(model, critorion, model_optim, epoch_idx):
    model.train()
    total_loss = 0
    start_time_epoch = time.time()
    for i, data in enumerate(train_loader):
        src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s, trg_raw = data
        src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s = wrap_variable(src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s)
        loss = model(src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        model_optim.step()
        model.zero_grad()
        total_loss += loss.detach()
        if i % args.print_every == 0 and i != 0:
            using_time = time.time() - start_time_epoch
            print('| ep %2d | %4d/%5d btcs | ms/btc %4.4f | loss %5.7f |' %(epoch_idx+1, i, len(train_loader), using_time * 1000 / (args.print_every), total_loss/args.print_every))
            total_loss = 0
            start_time_epoch = time.time()
    val_loss = infer(model, critorion)
    return val_loss, total_loss/len(train_loader)

def infer(model, critorion):
    model.eval()
    total_loss = 0
    for i, data in enumerate(val_loader):
        src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s ,trg_raw= data
        src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s = wrap_variable(src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s)
        loss = model(src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s)
        total_loss += loss.detach()
    return total_loss/len(val_loader)

def predict(model):
    model.eval()
    summ_list = []
    raw_list = []
    for i, data in enumerate(test_loader):
        src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s ,trg_raw = data
        src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s = wrap_variable(src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s)
        save_hyp = model.beam_predict(src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s)
        summ_list.append(save_hyp)
        raw_list.append(trg_raw)
    generate_summ(summ_list, raw_list)

def generate_summ(summ_list, tgt_seqs):
    id2token = data_loader.Dataset(args.pkl_path+"train.pkl").id2token
    summ_pred = []
    summ_raw = []
    for idxlist in summ_list:
        summ = [id2token[x] for x in idxlist if x != 0]
        strs = " ".join(summ)
        summ_pred.append(strs)
    for rawlists in tgt_seqs:
        tgt_summ = ""
        for rawlist in rawlists:
            for lists in rawlist:
                strs = " ".join(lists)
                tgt_summ += strs
        summ_raw.append(tgt_summ)


    for i in range(2):
        print("-------------pred summ-------------")
        print(summ_pred[i])
        print("-------------raw  summ-------------")
        print(summ_raw[i])




    ####to do

def eval_rouge(file):
    pass #to do

   


if __name__ == '__main__':
    main()



