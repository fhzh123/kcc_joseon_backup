from itertools import chain
from random import random, randrange

import torch
from torch.utils.data.dataset import Dataset


class HanjaKoreanDataset(Dataset):
    def __init__(self, hanja_list, korean_list, min_len=4, src_max_len=512, trg_max_len=512):
        hanja_list, korean_list = zip(*[(h, k) for h, k in zip(hanja_list, korean_list)\
            if min_len <= len(h) <= src_max_len and min_len <= len(k) <= trg_max_len])
        self.hanja_korean = [(h, k) for h, k in zip(hanja_list, korean_list)]
        self.hanja_korean = sorted(self.hanja_korean, key=lambda x: len(x[0])+len(x[1]))
        self.hanja_korean = self.hanja_korean[-1000:] + self.hanja_korean[:-1000]
        self.num_data = len(self.hanja_korean)
        
    def __getitem__(self, index):
        hanja, korean = self.hanja_korean[index]        
        return hanja, korean
    
    def __len__(self):
        return self.num_data


class HanjaDataset(Dataset):
    def __init__(self, hanja_list, additional_hanja, pad_idx=0, mask_idx=4, min_len=4, src_max_len=512):
        self.hanja_list = [h for h in hanja_list if min_len <= len(h) <= src_max_len]
        self.hanja_list = sorted(self.hanja_list, key=lambda x: len(x))
        self.hanja_list = self.hanja_list[-1000:] + self.hanja_list[:-1000]

        additional_hanja = [h for h in additional_hanja if min_len <= len(h) <= src_max_len]
        self.hanja_list.extend(additional_hanja)
        
        self.hanja_list = sorted(self.hanja_list, key=lambda x: len(x))
        self.hanja_list = self.hanja_list[-1000:] + self.hanja_list[:-1000]
        self.num_data = len(self.hanja_list)

        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        
    def __getitem__(self, index):
        rand_sentence = self.hanja_list[randrange(0, self.num_data)]
        sentence, label = self.get_random_tokens(rand_sentence)
        return sentence, label
    
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
            if p1 < 0.12:
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
    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences

        sentences_list = zip(*batch)
        sentences_list = [pack_sentence(sentences) for sentences in sentences_list]        
        return tuple(sentences_list)

    def __call__(self, batch):
        return self.pad_collate(batch)
