from random import randrange, random

import torch
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, hanja_list, korean_list, additional_hanja_list=None, 
                 pad_idx=0, mask_idx=4, min_len=4, src_max_len=512, trg_max_len=512):
        
        hanja_list, korean_list = zip(*[(h, k) for h, k in zip(hanja_list, korean_list)\
             if len(h) <= src_max_len and len(k) <= trg_max_len])
        self.hanja_korean = [(h, k) for h, k in zip(hanja_list, korean_list) \
            if len(h) >= min_len and len(k) >= min_len]
        self.hanja_korean = sorted(self.hanja_korean, key=lambda x: len(x[0])+len(x[1]))
        self.hanja_korean = self.hanja_korean[-1000:] + self.hanja_korean[:-1000]

        hanja_list = list(hanja_list)
        if additional_hanja_list is not None:
            additional_hanja_list = [h for h in additional_hanja_list \
                if min_len <= len(h) <= src_max_len]
            hanja_list.extend(additional_hanja_list)
        
        hanja_list = sorted(hanja_list, key=lambda x: len(x))
        hanja_list = hanja_list[-1000:] + hanja_list[:-1000]
        self.hanja_list = hanja_list
        self.num_data = len(self.hanja_korean)
        self.num_hanja_data = len(self.hanja_list)

        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        
    def __getitem__(self, index):
        hanja, korean = self.hanja_korean[index]
        rand_sentence = self.hanja_list[randrange(0, self.num_hanja_data)]
        masked_sentence, masked_label = self.get_random_tokens(rand_sentence)
        return hanja, korean, masked_sentence, masked_label
    
    def __len__(self):
        return self.num_data

    def get_random_tokens(self, sequence):
        masked_sentence = sequence.copy()
        output_label = [self.pad_idx for _ in range(len(sequence))]
        n, i = 0, 0
        while True:
            if i >= len(sequence):
                break

            p1 = random()
            if p1 < 0.15:
                # mask
                p2 = random()
                if p2 >= 0.9 and i < len(sequence) - 2:
                    # trigram
                    masked_sentence[i] = self.mask_idx
                    masked_sentence[i + 1] = self.mask_idx
                    masked_sentence[i + 2] = self.mask_idx
                    output_label[i] = sequence[i]
                    output_label[i + 1] = sequence[i + 1]
                    output_label[i + 2] = sequence[i + 2]
                    n += 3
                    i += 4
                elif 0.6 < p2 < 0.9 and i < len(sequence) - 1:
                    # bigram
                    masked_sentence[i] = self.mask_idx
                    masked_sentence[i + 1] = self.mask_idx
                    output_label[i] = sequence[i]
                    output_label[i + 1] = sequence[i + 1]
                    n += 2
                    i += 3
                else:
                    # unigram
                    masked_sentence[i] = self.mask_idx
                    output_label[i] = sequence[i]
                    n += 1
                    i += 2
            else:
                # mask input 말고 나머지는 pad index로 해서 training loss에서 제외하도록
                i += 1

        if n == 0:
            # 아무것도 마스킹 되지 않았으면 하나는 마스킹 하기
            rand_index = randrange(0, len(sequence))
            masked_sentence[rand_index] = self.mask_idx
            output_label[rand_index] = sequence[rand_index]
        
        return masked_sentence, output_label


class PadCollate:
    def __init__(self, pad_index=0, dim=0, src_max_len=None, trg_max_len=None):
        self.dim = dim
        self.pad_index = pad_index
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)
        
        (input_sequences, output_sequences, masked_sentences, masked_labels) = zip(*batch)
        batch_size = len(input_sequences)
        
        ### for input_items desc
        # find longest sequence
        if self.src_max_len is not None:
            input_seq_len = masked_seq_len = self.src_max_len
        else:
            input_seq_len = max(map(lambda x: len(x), input_sequences))
            masked_seq_len = max(map(lambda x: len(x), masked_sentences))
        
        # pad according to max_len
        input_sequences = [pad_tensor(torch.LongTensor(seq), input_seq_len, self.dim) for seq in input_sequences]
        input_sequences = torch.cat(input_sequences)
        input_sequences = input_sequences.view(batch_size, input_seq_len)

        ### for target_items desc
        if self.trg_max_len is not None:
            output_seq_len = self.trg_max_len
        else:
            output_seq_len = max(map(lambda x: len(x), output_sequences))
        
        # pad according to max_len
        output_sequences = [pad_tensor(torch.LongTensor(seq), output_seq_len, self.dim) for seq in output_sequences]
        output_sequences = torch.cat(output_sequences)
        output_sequences = output_sequences.view(batch_size, output_seq_len)

        # for masked sentence
        masked_sentences = [pad_tensor(torch.LongTensor(seq), masked_seq_len, self.dim) for seq in masked_sentences]
        masked_sentences = torch.cat(masked_sentences)
        masked_sentences = masked_sentences.view(batch_size, masked_seq_len)

        masked_labels = [pad_tensor(torch.LongTensor(seq), masked_seq_len, self.dim) for seq in masked_labels]
        masked_labels = torch.cat(masked_labels)
        masked_labels = masked_labels.view(batch_size, masked_seq_len)
        
        return input_sequences, output_sequences, masked_sentences, masked_labels

    def __call__(self, batch):
        return self.pad_collate(batch)