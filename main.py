import os
import argparse
import time

from dataloader import load_train_data, load_test_data, get_mapping, DataLoader
from model import Listener, Speller
from utils import seq2sen
from metric import word_error_rate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def save_checkpoint(listener, speller, path):
    listener_state = {
        'state_dict' : listener.state_dict()
    }

    speller_state = {
        'state_dict' : speller.state_dict()
    }
    
    torch.save(listener_state, path + 'listener')
    torch.save(speller_state, path + 'speller')
    print('A check point has been generated : ' + path)

def main(args):
    sos_idx = 0
    eos_idx = 1
    pad_idx = -1
    pad_val = 0.0

    feature_dim = 40
    listener_hidden_dim = 256
    num_of_pyramidal_layers = 3
    speller_hidden_dim = 512
    attention_hidden_dim = 128
    num_of_classes = 30
    max_label_len = 300

    learning_rate = 0.2
    geometric_decay = 0.98

    device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")
    listener = Listener(feature_dim, listener_hidden_dim, num_of_pyramidal_layers).to(device)
    speller = Speller(speller_hidden_dim, listener_hidden_dim, attention_hidden_dim, num_of_classes, device).to(device)
    #print(device, listener, speller)

    if not args.test:
        # train
        src, trg = load_train_data(args.path)
        train_loader = DataLoader(src, trg, args.batch_size, pad_idx, pad_val)

        criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
        optimizer = torch.optim.ASGD([{'params':listener.parameters()}, {'params':speller.parameters()}], lr=learning_rate)

        print('Start training ...')
        for epoch in range(args.epochs):
            start_epoch = time.time()
            i = 0

            for src_batch, tgt_batch in train_loader:
                batch_start = time.time()

                src_batch = torch.tensor(src_batch).to(device)
                trg_batch = torch.tensor(tgt_batch).to(device)
                
                trg_input = trg_batch[:,:-1] 
                trg_output = trg_batch[:,1:].contiguous().view(-1)
                
                h = listener(src_batch)
                preds = speller(trg_input, h)
                
                # lr decay for every 1/20 epoch
                if (i+1) % ((train_loader.size//args.batch_size)//20) is 0 :
                    learning_rate = geometric_decay * learning_rate
                    print('learing rate decayed : %.4f'%(learning_rate))
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate

                optimizer.zero_grad()

                loss = criterion(preds.view(-1, preds.size(-1)), trg_output)
                loss.backward()

                optimizer.step()

                i = i+1
                
                # flush the GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_time = time.time() - batch_start
                print('[%d/%d][%d/%d] train loss : %.4f | time : %.2fs'%(epoch+1, 100, i, train_loader.size//args.batch_size + 1, loss.item(), batch_time))
                
            epoch_time = time.time() - start_epoch
            print('Time taken for %d epoch : %.2fs'%(epoch+1, epoch_time))

            save_checkpoint(listener, speller, 'checkpoints/epoch_%d_'%(epoch+1))

        print('End of the training')
        save_checkpoint(listener, speller, 'checkpoints/final_')
    else:
        if os.path.exists(args.checkpoint + 'listener') and os.path.exists(args.checkpoint + 'speller'):
            listener_checkpoint = torch.load(args.checkpoint + 'listener')
            listener.load_state_dict(listener_checkpoint['state_dict'])
            print("trained model " + args.checkpoint + "listener is loaded")

            speller_checkpoint = torch.load(args.checkpoint + 'speller')
            speller.load_state_dict(speller_checkpoint['state_dict'])
            print("trained model " + args.checkpoint + "speller is loaded")

        # test
        src, trg = load_test_data(args.path)
        test_loader = DataLoader(src, trg, args.batch_size, pad_idx, pad_val)
        mapping = get_mapping(args.path)

        j = 0
        pred = []
        for src_batch, trg_batch in test_loader:
            # predict pred_batch from src_batch with your model.
            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]

            src_batch = torch.tensor(src_batch).to(device)
            trg_batch = torch.tensor(trg_batch).to(device)
            
            max_length = trg_batch.size(1)
            
            pred_batch = torch.zeros(args.batch_size, 1, dtype = int).to(device) # [batch, 1] = [[0],[0],...,[0]]
            
            # eos_mask[i] = 1 means i-th sentence has eos
            eos_mask = torch.zeros(args.batch_size, dtype = int)
            
            h = listener(src_batch)
            
            for k in range(max_length):
                start = time.time()
                output = speller(pred_batch, h) # [batch, k+1, num_class]

                # greedy search
                output = torch.argmax(F.softmax(output, dim = -1), dim = -1) # [batch_size, k+1]
                predictions = output[:,-1].unsqueeze(1)
                pred_batch = torch.cat([pred_batch, predictions], dim = -1)

                for i in range(args.batch_size):
                    if predictions[i] == eos_idx:
                        eos_mask[i] = 1

                # every sentence has eos
                if eos_mask.sum() == args.batch_size :
                    break
                    
                t = time.time() - start
                print("[%d/%d][%d/%d] prediction done | time : %.2fs"%(j, test_loader.size // args.batch_size + 1, k+1, max_length, t))
            j += 1

            # flush the GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[%d/%d] prediction done"%(j, test_loader.size // batch_size + 1))
            pred += seq2sen(pred_batch.cpu().numpy().tolist(), mapping)
            ref += seq2sen(trg_batch.cpu().numpy().tolist(), mapping)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        with open('results/ref.txt', 'w') as f:
            for line in ref:
                f.write('{}\n'.format(line))

        WER = word_error_rate(ref, pred)
        print("Test End : WER %.2f%%"%(WER))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAS')
    parser.add_argument(
        '--path',
        type=str,
        default='data/libri_fbank40_char30')

    parser.add_argument(
        '--epochs',
        type=int,
        default=200)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16)

    parser.add_argument(
        '--test',
        action='store_true')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint/final_'
    )

    args = parser.parse_args()

    main(args)
