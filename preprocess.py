from __future__ import (division, absolute_import, print_function, unicode_literals)
import os
import numpy as np
import gzip
import os.path
import logging
import glob
import pickle

from tqdm import tqdm
import codecs
import re
import math
from collections import defaultdict
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
    from io import open

def read_file(file):
    with open(file, "r", encoding='utf-8') as f:
        content = f.read().strip().split('\n\n')
        doc = content[1].strip().split("<eos>")
        raw_tokens_doc = [sent.strip().split() + ["<eos>"] for sent in doc[:-1]]
        raw_tokens_doc += ["<eod>"]
        entitys = content[2].rstrip().split("\n")
        summs = content[3].rstrip().split("\n")
        raw_token_summ = [sent.strip().split() for sent in summs]
        Max_lens_doc = max([len(x) for x in raw_tokens_doc])
        Max_lens_summ = max([len(x) for x in raw_token_summ])
        doc_summs_dict = {"doc_tokens": raw_tokens_doc,
                          "summ_tokens": raw_token_summ,
                          "maxlens_doc": Max_lens_doc,
                          "maxlens_summ": Max_lens_summ,
                          "num_sent_doc": len(raw_tokens_doc),
                          "num_sent_summ": len(raw_token_summ)}
    return doc_summs_dict

def wordNormalize(word):
    word = word.lower()
    #word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)
    word = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}$", 'DATE_TOKEN', word)
    word = re.sub("[0-9]{1,2}[.:]+[0-9]{2}:[0-9]{2}$", 'TIME_TOKEN', word)
    word = re.sub("[0-9]{2}:[0-9]{2}$", 'TIME_TOKEN', word)
    word = re.sub("^[0-9]+[.,//-]*[0-9]*$", 'NUMBER_TOKEN', word)
    return word

def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing

def create_mapppings(data_list):
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>", "<sod>", "<eod>"]
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    token_frq_dict = defaultdict(int)

    caseing2id = {entries[idx]:idx for idx in range(len(entries))}
    for token in special_tokens:
        token_frq_dict[token] = 1
        
    for data in data_list:
        for part in data.values():
            if isinstance(part, list):
                for sents in part:
                    for word in sents:
                        token = wordNormalize(word)
                        token_frq_dict[token] += 1
    mappings = {"token_frq_dict": token_frq_dict,
                "caseing2id": caseing2id}
    return mappings


def build_emb_matrix(path, mappings, dim, unk_frequence=50):
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>", "<sod>", "<eod>"]
    token2id, id2token = {}, {}
    embeddingslist = []
    token2id["<pad>"] = len(token2id)
    id2token[len(id2token)] = "<pad>"
    vector = np.zeros(dim)
    embeddingslist.append(vector)

    stdv = 3.0 / math.sqrt(dim)
    for token in special_tokens[1:]:
        token2id[token] = len(token2id)
        id2token[len(id2token)] = token
        vector = np.random.uniform(-stdv, stdv, dim)
        embeddingslist.append(vector)

    token_frq_dict = mappings["token_frq_dict"]
    print("Reading embeddings files...")
    with codecs.open(path, "rb", encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            split = line.rstrip().split(" ")
            token = split[0]
            assert len(vector) == dim
            if token in token_frq_dict:
                token2id[token] = len(token2id) 
                id2token[len(id2token)] = token
                vector = np.array([float(num) for num in split[1:]])
                embeddingslist.append(vector)
    print("Deal with unk tokens out of pretrained embeddings...")
    count = 0
    for key, value in tqdm(token_frq_dict.items()):
        if key not in token2id:
            if value >= unk_frequence:
                token2id[key] = len(token2id)
                id2token[len(id2token)] = key
                vector = np.random.uniform(-stdv, stdv, dim)
                embeddingslist.append(vector)
            else:
                token2id[key] = token2id["<unk>"]
                id2token[len(id2token)] = key
                embeddingslist.append(np.zeros(dim))
                count += 1
    print("%d tokens in voc are mapped to <unk>" %count)
    embeddings = np.array([embeddingslist])
    mappings["token2id"] = token2id
    mappings["id2token"] = id2token
    return mappings, embeddings

def add_token2idx_list(data_list, mappings):
    token2id = mappings["token2id"]
    id2token = mappings["id2token"]
    for i in range(len(data_list)):
        for key, value in data_list[i].items():
            if key == "doc_tokens":
                doc_idx = [[token2id[wordNormalize(token)] for token in x] for x in value]
            if key == "summ_tokens":
                summs_idx = [[token2id[wordNormalize(token)] for token in x] for x in value]
        data_list[i]["doc_idx"] = doc_idx
        data_list[i]["summs_idx"] = summs_idx

    return data_list

def build_data_list(corp_dir):
    fps = glob.glob(corp_dir + "*.fina")
    data_list = []

    for fp in tqdm(fps):
        doc_summs_dict = read_file(fp)
        data_list.append(doc_summs_dict)

    return data_list

def create_pkl(corp_dir, pkl_path, path_emb, dim=300, unk_frequence=50):
    print("Reading raw files...")
    if os.path.isfile(pkl_path+"train.pkl"):
        file_name = os.path.split(pkl_path)[1]
        print("%s file exists." %file_name)
    else:
        corpus = {}
        str_list = ["train/", "val/", "test/"]
        data_lists = []
        for item in str_list:
            corp_path = corp_dir + item
            data_list = build_data_list(corp_path)
            data_lists.append(data_list)

        mappings = create_mapppings(data_lists[0]+data_lists[1]+data_lists[2])
        mappings, embeddings = build_emb_matrix(path_emb, mappings, dim, unk_frequence)
        train_data_list = add_token2idx_list(data_lists[0], mappings)
        val_data_list = add_token2idx_list(data_lists[1], mappings)
        test_data_list = add_token2idx_list(data_lists[2], mappings)
        
        corpus_train = {"mappings": mappings,
                        "data_list": train_data_list}
        corpus_val = {"mappings": mappings,
                        "data_list": val_data_list}
        corpus_test = {"mappings": mappings,
                        "data_list": test_data_list}
        save_pkl(pkl_path+"train.pkl", corpus_train)
        save_pkl(pkl_path+"val.pkl", corpus_val)
        save_pkl(pkl_path+"test.pkl", corpus_test)
        save_pkl(pkl_path+"embeddings.pkl", embeddings)

def save_pkl(path, corpus):
    path_ = os.path.split(path)[0]
    if not os.path.isdir(path_):
        os.mkdir(path_)
    with open(path, "wb") as f:
        pickle.dump(corpus, f)

def read_pkl(path):
    with open(path, "rb") as f:
        corpus = pickle.load(f)
    return corpus



def test():
    dir_path = "./data/"
    save_path = "./data/pkl/"
    path_emb = "/u/lupeng/Project/code/vqvae_kb/.vector_cache/glove.42B.300d.txt"
    create_pkl(dir_path, save_path, path_emb)
    corpus = read_pkl(save_path+"train.pkl")
    print(type(corpus))
    print(len(corpus["mappings"]["token_frq_dict"]))

if __name__ == '__main__':
    test()

