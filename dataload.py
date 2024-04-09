#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：dataset.py
@File    ：dataload.py
@Author  ：Polaris
@Date    ：2022-05-12 13:"""

from torch.utils.data import Dataset, DataLoader
from model import tokenizer


def read_data_from_csv(filename):
    data_dic_lis_train = []
    with open(filename + '/train_in.txt', "r", encoding="utf-8") as fin:
        ind = 0
        fout = open(filename + '/train_out.txt', "r", encoding="utf-8")
        ind = 0
        for iin, iout in zip(fin, fout):
            ind += 1
            iin = iin[0:-1].split()
            iout = iout[0:-1].split()
            iin.append('。')
            iout.append('。')
            data_dic_lis_train.append(iin+iout)
            ind += 1
        fout.close()

    data_dic_lis_test = []
    with open(filename + '/test_in.txt', "r", encoding="utf-8") as fin:
        ind = 0
        fout = open(filename + '/tset_out.txt', "r", encoding="utf-8")
        for iin, iout in zip(fin, fout):
            iin = iin[0:-1].split()
            iout = iout[0:-1].split()
            iin.append('。')
            iout.append('。')
            data_dic_lis_test.append(iin+iout)
            ind += 1
        fout.close()

    return data_dic_lis_train, data_dic_lis_test


class NERset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.datasize = len(dataset)

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        return self.dataset[index]


def coffate_fn(examples):
    sents = []
    for sent in examples:
        sents.append(sent)
    tokenized_inputs = tokenizer(sents,
                                 truncation=True,
                                 padding=True,
                                 # return_offsets_mapping=True,
                                 is_split_into_words=True,
                                 max_length=512,
                                 return_tensors="pt")
    return tokenized_inputs


def loadset(filename, batch_size=32):
    traindata, testdata = read_data_from_csv(filename)
    trainset = NERset(traindata)
    testset = NERset(testdata)
    train_dataloader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  collate_fn=coffate_fn,
                                  shuffle=True)
    test_dataloader = DataLoader(testset,
                                 batch_size=batch_size,
                                 collate_fn=coffate_fn,
                                 shuffle=True)
    return train_dataloader, test_dataloader