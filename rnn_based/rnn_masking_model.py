# Import Modules
import math
import random
import numpy as np

# Import PyTorch
import torch
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch import nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, pad_idx=0, dropout=0.0, embedding_dropout=0.0):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=n_layers, bidirectional=True,
                          layer_norm=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=pad_idx)  # embedding layer

    def forward(self, src, hidden=None, cell=None):
        # Source sentence embedding
        embeddings = self.embedding(src)  # (max_caption_length, batch_size, embed_dim)
        embedded = F.dropout(embeddings, p=self.embedding_dropout,
                             inplace=True)  # (max_caption_length, batch_size, embed_dim)
        # Bidirectional GRU
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = torch.tanh(self.linear(outputs))  # (max_caption_length, batch_size, embed_dim)
        return outputs, hidden