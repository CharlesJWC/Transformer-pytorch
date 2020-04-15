#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Attention is all you need" Implementation
20193640 Jungwon Choi
'''
import torch
# from torchnlp.datasets import wmt_dataset, multi30k_dataset
from torchtext.datasets import WMT14, Multi30k
from torchtext.data import Field, BucketIterator

import numpy as np
import os

#===============================================================================
''' General MNT Dataloader for inheritance '''
class MNT_Dataloader():
    #===========================================================================
    ''' Translate idx list to sentence '''
    def translate_sentence(self, idx_list, type='trg', string=False):
        word_list=[]
        for idx in idx_list[1:]:
            if type == 'src':
                word = self.SRC.vocab.itos[idx]
            else:
                word = self.TRG.vocab.itos[idx]
            if word != "<eos>":
                word_list.append(word)
                continue
            break
        if string:
            return [' '.join(word_list)]
        else:
            return word_list
    #===========================================================================
    ''' Get train dataset loader '''
    def get_train_loader(self, batch_size=64):
        train_loader = BucketIterator(self.train_data,
                                        batch_size=batch_size,
                                        train=True,
                                        shuffle=True)
        return train_loader

    #===========================================================================
    ''' Get validation dataset loader '''
    def get_val_loader(self, batch_size=64):
        val_loader = BucketIterator(self.val_data,
                                        batch_size=batch_size,
                                        train=False,
                                        shuffle=False)
        return val_loader

    #===========================================================================
    ''' Get test dataset loader '''
    def get_test_loader(self, batch_size=64):
        test_loader = BucketIterator(self.test_data,
                                        batch_size=batch_size,
                                        train=False,
                                        shuffle=False)
        return test_loader

#===============================================================================
''' WMT2014 Dataloader maker '''
class WMT2014_Dataloader(MNT_Dataloader):
    #===========================================================================
    ''' Initialization '''
    def __init__(self):
        # Create Source and Target Tokenizer
        self.SRC = Field(tokenize='spacy',      # Tokenize based on spacy
                    tokenizer_language="en",    # Set Src language
                    init_token='<sos>',         # init of sentence token
                    eos_token='<eos>',          # end of sentence token
                    lower=True,                 # Lower the character
                    batch_first=True)           # Make setence as same length
        self.TRG = Field(tokenize='spacy',      # Tokenize based on spacy
                    tokenizer_language="de",    # Set Src language
                    init_token='<sos>',         # init of sentence token
                    eos_token='<eos>',          # end of sentence token
                    lower=True,                 # Lower the character
                    batch_first=True)           # Make setence as same length

        # Load and split dataset
        self.train_data, self.val_data, self.test_data = WMT14.splits(
                                                    exts=('.en', '.de'),
                                                    fields=(self.SRC, self.TRG),
                                                    root='data')

        # Build vocaburary based on train dataset
        self.SRC.build_vocab(self.train_data, min_freq=2)
        self.TRG.build_vocab(self.train_data, min_freq=2)

        # Get pad and sos idx
        self.pad_idx = self.SRC.vocab.stoi['<pad>']
        self.sos_idx = self.SRC.vocab.stoi['<sos>']

#===============================================================================
''' multi30k Dataloader maker '''
class Multi30k_Dataloader(MNT_Dataloader):
    #===========================================================================
    ''' Initialization '''
    def __init__(self):
        # Create Source and Target Tokenizer
        self.SRC = Field(tokenize='spacy',      # Tokenize based on spacy
                    tokenizer_language="de",    # Set Src language
                    init_token='<sos>',         # init of sentence token
                    eos_token='<eos>',          # end of sentence token
                    lower=True,                 # Lower the character
                    batch_first=True)           # Make setence as same length
        self.TRG = Field(tokenize='spacy',      # Tokenize based on spacy
                    tokenizer_language="en",    # Set Src language
                    init_token='<sos>',         # init of sentence token
                    eos_token='<eos>',          # end of sentence token
                    lower=True,                 # Lower the character
                    batch_first=True)           # Make setence as same length

        # Load and split dataset
        self.train_data, self.val_data, self.test_data = Multi30k.splits(
                                                    exts=('.de', '.en'),
                                                    fields=(self.SRC, self.TRG),
                                                    root='data')

        # Build vocaburary based on train dataset
        self.SRC.build_vocab(self.train_data, min_freq=2)
        self.TRG.build_vocab(self.train_data, min_freq=2)

        # Get pad and sos idx
        self.pad_idx = self.SRC.vocab.stoi['<pad>']
        self.sos_idx = self.SRC.vocab.stoi['<sos>']


#===============================================================================
''' Check the maximun length '''
if __name__ == '__main__':
    dataloader = Multi30k_Dataloader()

    #===========================================================================
    # For train dataset
    train_loader = dataloader.get_train_loader(64)
    src_max_size = 0
    trg_max_size = 0
    for batch in train_loader:
        srcs, trgs = batch.src, batch.trg
        if srcs.size(1) > src_max_size:
            src_max_size = srcs.size(1)
        if trgs.size(1) > trg_max_size:
            trg_max_size = srcs.size(1)
    print('Train:', src_max_size, trg_max_size)

    #===========================================================================
    # For validation dataset
    val_loader = dataloader.get_val_loader(64)
    src_max_size = 0
    trg_max_size = 0
    for batch in val_loader:
        srcs, trgs = batch.src, batch.trg
        if srcs.size(1) > src_max_size:
            src_max_size = srcs.size(1)
        if trgs.size(1) > trg_max_size:
            trg_max_size = srcs.size(1)
    print('Valid:', src_max_size, trg_max_size)

    #===========================================================================
    # For test dataset
    test_loader = dataloader.get_test_loader(64)
    src_max_size = 0
    trg_max_size = 0
    for batch in test_loader:
        srcs, trgs = batch.src, batch.trg
        if srcs.size(1) > src_max_size:
            src_max_size = srcs.size(1)
        if trgs.size(1) > trg_max_size:
            trg_max_size = srcs.size(1)
    print('Test :', src_max_size, trg_max_size)
