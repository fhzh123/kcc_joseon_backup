# Import Module
import os
import time
import pickle
import argparse
import math
import warnings
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.utils.data import DataLoader


# Import Custom Module
# from dataset import CustomDataset, PadCollate
from rnn_masking_model import Model

with open('C:/Users/cys40/translation/preprocessed_data.pkl','rb') as pickle_file:
    data = pickle.load(pickle_file)


def main(args):
    # Setting
    warnings.simplefilter("ignore", UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loading
    print('Data loading and data spliting...')
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
        src_word2id = data['hanja_word2id']
        src_vocab = [k for k in src_word2id.keys()]
        trg_word2id = data['korean_word2id']
        trg_vocab = [k for k in trg_word2id.keys()]
        train_src_list = data['train_hanja_indices']
        train_add_hanja = data['train_additional_hanja_indices']
        valid_src_list = data['valid_hanja_indices']
        valid_add_hanja = data['valid_additional_hanja_indices']

        src_vocab_num = len(src_vocab)
        trg_vocab_num = len(trg_vocab)

        del data
    print('Done!')

    # DataLoader Setting
    h_dataset = {
        'train': HanjaDataset(train_src_list, train_add_hanja, pad_idx=0, mask_idx=4,
                              min_len=4, src_max_len=150),
        'valid': HanjaDataset(valid_src_list, valid_add_hanja, pad_idx=0, mask_idx=4,
                              min_len=4, src_max_len=150)

    }
    h_loader = {
        'train': DataLoader(h_dataset['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=4),
        'valid': DataLoader(h_dataset['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=4),
    }

    encoder = Encoder(src_vocab_num, 256, 256, n_layers=6,
                      pad_idx=0, dropout=0.3, embedding_dropout=0.2)
    encoder_linear = nn.Linear(256, trg_vocab_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    encoder.to(device)
    encoder_linear.to(device)
    for e in range(args.num_epoch):
        for phase in ['train', 'valid']:
            if phase == 'train':
                encoder.train()
            if phase == 'valid':
                encoder.eval()
                val_loss = 0
            total_loss_list = list()
        for src, trg in h_loader['train']:
            src.transpose(0, 1).to(device)
            masked_position = trg != 0

            optimizer.zero_grad()

            # Encoder Output
            encoder_out, hidden = encoder(src)
            encoder_out = encoder_out.transpose(0, 1)
            encoder_out = encoder_linear(encoder_out[masked_position])

            # loss
            masked_label = trg[masked_position]
            loss = criterion(predicted, masked_labels_)
            loss.backward()
            optimizer.step()
            if phase == 'train':
                loss.backward()
                optimizer.step()
                freq += 1
                if freq == args.print_freq:
                    total_loss = loss.item()
                    print("[loss:%5.2f][pp:%5.2f]" % (total_loss, math.exp(total_loss)))
                    total_loss_list.append(total_loss)
                    freq = 0
        if phase == 'train':
            pd.DataFrame(total_loss_list).to_csv('./rnn_based/save/{} epoch_loss.csv'.format(e), index=False)
        if phase == 'valid':
            val_loss /= len(h_loader['valid'])
            print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS | spend_time:%5.2fmin"
                  % (e, val_loss, math.exp(val_loss), (time.time() - start_time_e) / 60))
            if not best_val_loss or val_loss < best_val_loss:
                print("[!] saving model...")
        scheduler.step()

if __name__ == '__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='SJW argparser')
    parser.add_argument('--data_path',
        default='C:/Users/cys40/translation/preprocessed_data.pkl',
        type=str, help='path of data pickle file (train)')
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='Preprocessed vocabulary size; Default is 32000')
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
    parser.add_argument('--n_layers', type=int, default=5, help='Model layers; Default is 5')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Ratio; Default is 0.5')
    parser.add_argument('--embedding_dropout', type=float, default=0.3, help='Embedding Dropout Ratio; Default is 0.3')

    parser.add_argument('--print_freq', type=int, default=100, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()

    main(args)