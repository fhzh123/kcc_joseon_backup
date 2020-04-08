# Import Module
import os
import time
import pickle
import argparse
import math
import warnings
import numpy as np
import pandas as pd
import sentencepiece as spm

from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.utils.data import DataLoader

# Import Custom Module
from dataset import HanjaKoreanDataset, PadCollate
from module import Encoder, Decoder, Seq2Seq

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Data loading and data spliting...')
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
        src_word2id = data['hanja_word2id']
        src_vocab = [k for k in src_word2id.keys()]
        trg_word2id = data['korean_word2id']
        trg_vocab = [k for k in trg_word2id.keys()]
        train_src_list = data['train_hanja_indices']
        train_trg_list = data['train_korean_indices']
        valid_src_list = data['valid_hanja_indices']
        valid_trg_list = data['valid_korean_indices']

        src_vocab_num = len(src_vocab)
        trg_vocab_num = len(trg_vocab)

        del data
    print('Done!')

    # Dataset & Dataloader setting
    dataset_dict = {
        'train': HanjaKoreanDataset(train_src_list, train_trg_list, min_len=args.min_len, 
                                    src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'valid': HanjaKoreanDataset(valid_src_list, valid_trg_list, min_len=args.min_len, 
                                    src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # Model Setting
    print("Instantiating models...")
    encoder = Encoder(src_vocab_num, args.embed_size, args.hidden_size, n_layers=args.n_layers, 
                    pad_idx=args.pad_idx, dropout=args.dropout, embedding_dropout=args.embedding_dropout)
    decoder = Decoder(args.embed_size, args.hidden_size, trg_vocab_num, n_layers=args.n_layers, 
                    pad_idx=args.pad_idx, dropout=args.dropout, embedding_dropout=args.embedding_dropout)
    seq2seq = Seq2Seq(encoder, decoder, device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, seq2seq.parameters()), lr=args.lr, weight_decay=args.w_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    #criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    torch_utils.clip_grad_norm_(seq2seq.parameters(), args.grad_clip)
    seq2seq.to(device)
    print(seq2seq)

    print('Model train start...')
    best_val_loss = None
    if not os.path.exists('../rnn_based/save'):
        os.mkdir('../rnn_based/save')
    for e in range(args.num_epoch):
        start_time_e = time.time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                seq2seq.train()
            if phase == 'valid':
                seq2seq.eval()
                val_loss = 0
            total_loss_list = list()
            freq = args.print_freq - 1
            for src,trg in tqdm(dataloader_dict[phase]):
                # Sourcen, Target sentence setting
                src = src.transpose(0,1).to(device)
                trg = trg.transpose(0,1).to(device)

                # Optimizer setting
                optimizer.zero_grad()

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    teacher_forcing_ratio = 0.5 if phase=='train' else 0
                    output = seq2seq(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
                    output_flat = output[1:].view(-1, trg_vocab_num)
                    trg_flat = trg[1:].contiguous().view(-1)
                    #loss = criterion(output_flat, trg_flat)
                    loss = F.cross_entropy(output[1:].transpose(0,1).contiguous().view(-1, trg_vocab_num), 
                                        trg[1:].transpose(0,1).contiguous().view(-1), ignore_index=args.pad_idx)
                    if phase == 'valid':
                        val_loss += loss.item()

                # If phase train, then backward loss and step optimizer and scheduler
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    # Print loss value only training
                    freq += 1
                    if freq == args.print_freq:
                        total_loss = loss.item()
                        print("[loss:%5.2f][pp:%5.2f]" % (total_loss, math.exp(total_loss)))
                        total_loss_list.append(total_loss)
                        freq = 0
                        
            # Finishing iteration
            if phase == 'train':
                pd.DataFrame(total_loss_list).to_csv('../rnn_based/save/{} epoch_loss.csv'.format(e), index=False)
            if phase == 'valid': 
                val_loss /= len(dataloader_dict['valid'])
                print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS | spend_time:%5.2fmin"
                        % (e, val_loss, math.exp(val_loss), (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss < best_val_loss:
                    print("[!] saving model...")
                    torch.save(seq2seq.state_dict(), '../rnn_based/save/seq2seq_{}.pt'.format(e))
                    best_val_loss = val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joseon NMT argparser')
    parser.add_argument('--data_path', 
        default='../../preprocessing/preprocessed_data.pkl', 
        type=str, help='path of data pickle file (train)')
    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')
    parser.add_argument('--mask_idx', default=4, type=int, help='index of mask token')

    parser.add_argument('--min_len', type=int, default=4, help='Minimum Length of Sentences; Default is 4')
    parser.add_argument('--src_max_len', type=int, default=300, help='Max Length of Source Sentence; Default is 300')
    parser.add_argument('--trg_max_len', type=int, default=384, help='Max Length of Target Sentence; Default is 384')

    parser.add_argument('--num_epoch', type=int, default=10, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size; Default is 8')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate; Default is 1e-4')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')

    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden State Vector Dimension; Default is 256')
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding Vector Dimension; Default is 256')
    parser.add_argument('--n_layers', type=int, default=6, help='Model layers; Default is 6')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Ratio; Default is 0.3')
    parser.add_argument('--embedding_dropout', type=float, default=0.2, help='Embedding Dropout Ratio; Default is 0.2')

    parser.add_argument('--print_freq', type=int, default=500, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()

    main(args)