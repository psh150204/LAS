import os
import pickle
import numpy as np

def get_mapping(data_root):
    with open(data_root + '/mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    
    return mapping

def load_train_data(data_root):
    src_train_path = data_root + '/train-clean-100/'
    trg_train_path = data_root + '/train-clean-100.csv'
    
    src_train, trg_train = [], []
    
    # load trg_train
    lines = []
    with open(trg_train_path, 'r') as reader:
        for line in reader:
            fields = line.split(',')
            lines.append(fields)

    # exclude the first line
    lines = lines[1:]
    # filenames are sorted by the sequence length
    train_files = [line[1] for line in lines]
    
    for line in lines:
        label = list(map(int, line[2].split('_')))
        trg_train.append(label)
    
    # load src train
    for name in train_files:
        src_train.append(np.load(data_root + '/' + name))
        
    return src_train, trg_train

def load_test_data(data_root):
    src_test_path = data_root + '/test-clean/'
    trg_test_path = data_root + '/test-clean.csv'
    
    src_test, trg_test = [], []
    
    # load trg_test
    lines = []
    with open(trg_test_path, 'r') as reader:
        for line in reader:
            fields = line.split(',')
            lines.append(fields)

    # exclude the first line
    lines = lines[1:]
    # filenames are sorted by the sequence length
    test_files = [line[1] for line in lines]

    for line in lines:
        label = list(map(int, line[2].split('_')))
        trg_test.append(label)
     
    # load src_test
    for name in test_files:
        src_test.append(np.load(data_root + '/' + name))
    
    return src_test, trg_test

class DataLoader:
    def __init__(self, src, tgt, batch_size, pad_idx, pad_val):
        assert len(src) == len(tgt), 'Number of sentences in source and target are different.'
        self.src = src
        self.tgt = tgt
        self.size = len(src)
        self.batch_size = batch_size
        self.pad_idx = pad_idx
        self.pad_val = pad_val

    def __iter__(self):
        self.index = 0
        return self

    
    def src_pad(self, batch):
        max_len = 0
        for seq in batch:
            if max_len < len(seq):
                max_len = len(seq)
        
        remainder = max_len % 8
        max_len += 0 if remainder is 0 else 8 - remainder
        
        for i in range(len(batch)):
            if max_len - len(batch[i]) is not 0:
                padding = np.array([[self.pad_val] * 40] * (max_len - len(batch[i])))
                batch[i] = np.concatenate((np.array(batch[i]), padding), axis = 0).tolist()

        return batch
    
    
    def trg_pad(self, batch):
        max_len = 0
        for seq in batch:
            if max_len < len(seq):
                max_len = len(seq)

        for i in range(len(batch)):
            batch[i] += [self.pad_idx] * (max_len - len(batch[i]))

        return batch
    

    def __next__(self):
        if self.batch_size * self.index >= self.size:
            raise StopIteration

        src_batch = self.src[self.batch_size * self.index : self.batch_size * (self.index+1)]
        tgt_batch = self.tgt[self.batch_size * self.index : self.batch_size * (self.index+1)]

        src_batch = self.src_pad(src_batch)
        tgt_batch = self.trg_pad(tgt_batch)

        self.index += 1

        return src_batch, tgt_batch