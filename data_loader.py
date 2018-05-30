import nltk
import torch
import torch.utils.data as data
import pickle
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from preprocess import read_pkl



class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, pkl_path):
        """Reads source and target sequences from pkl files."""
        self.corpus = read_pkl(pkl_path)
        self.datalist = self.corpus["data_list"]

        self.token2id = self.corpus["mappings"]["token2id"]
        self.id2token = self.corpus["mappings"]["id2token"]

        self.num_total_pairs = len(self.datalist)


    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.datalist[index]["doc_idx"]
        trg_seq = self.datalist[index]["summs_idx"]
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_pairs

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        src_mask_w: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        src_mask_s: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        tgt_seqs: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        tgt_mask_w: torch tensor of shape (batch_size, padded_num_sent, padded_length).
        tgt_mask_s: torch tensor of shape (batch_size, padded_num_sent, padded_length).
    """
    def merge(sequences):
        num_sents = [len(sent) for sent in sequences]
        seq_lens = [[len(seq) for seq in sent] for sent in sequences]
        max_length_seq = max([max([len(seq) for seq in sent]) for sent in sequences])

        padded_seqs = torch.zeros(len(sequences), max(num_sents), max_length_seq).long()
        mask_w = torch.zeros(len(sequences), max(num_sents), max_length_seq).long()
        mask_s = torch.zeros(len(sequences), max(num_sents)).long()
        padded_seqs[:, :, 0] = 3
        mask_w[:, :, 0] = 1
        for i, sent in enumerate(sequences):
            end = num_sents[i]
            mask_s[i, :end] = 1
            for j, seq in enumerate(sent):
                endd = seq_lens[i][j]
                padded_seqs[i, j, :endd] = torch.LongTensor(seq[:endd])
                mask_w[i, j, :endd] = 1

        return padded_seqs, mask_w, mask_s

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, tgt_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_mask_w, src_mask_s = merge(src_seqs)
    tgt_seqs, tgt_mask_w, tgt_mask_s = merge(tgt_seqs)

    return src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s
def split_data_loader(pkl_path, batch_size, val_num, test_num):
    ## define our indices -- our dataset has 9 elements and we want a 8:4 split
    mydset = Dataset(pkl_path)
    num_train = len(mydset)
    indices = list(range(num_train))

    # Random, non-contiguous split
    np.random.seed(123321)
    valandtest_idx = np.random.choice(indices, size=val_num+test_num, replace=False)
    train_idx = list(set(indices) - set(valandtest_idx[: val_num]))
    val_idx = list(set(valandtest_idx[: val_num]))
    test_idx = list(set(valandtest_idx[val_num:]))

    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(mydset, 
                                               batch_size=batch_size,
                                               collate_fn=collate_fn,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(mydset, 
                                             batch_size=batch_size,
                                             collate_fn=collate_fn, 
                                             sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(mydset, 
                                              batch_size=batch_size, 
                                              collate_fn=collate_fn,
                                              sampler=test_sampler)

    return train_loader, val_loader, test_loader


def get_loader(pkl_path, batch_size=1):
    """Returns data loader for custom dataset.

    Args:
        pkl_path: pkl file path for source domain.
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(pkl_path)

    # data loader for custome dataset
    # this will return (src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

    return data_loader

def ok():
    pass
def test():
    pkl_path = "./data/pkl/data.pkl"
    a, b, c = split_data_loader(pkl_path, 1, val_num= 10, test_num=30)
    print(len(a))
    print(len(b))
    print("OK")
    #src_seqs, src_mask_w, src_mask_s, tgt_seqs, tgt_mask_w, tgt_mask_s = next(data_iter)
    #print(src_seqs.size())

if __name__ == '__main__':
    test()