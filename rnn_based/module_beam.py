import math
import torch
import random
import operator
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from sru import SRU
import numpy as np
from queue import PriorityQueue, Queue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, word2vec,
                 n_layers=1, padding_index=3, dropout=0.0, embedding_dropout=0.0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.wv = word2vec.wv
        self.gru = SRU(embed_size, hidden_size, num_layers=n_layers, bidirectional=True,
                       layer_norm=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        # Word2Vec rescale
        self.wv = word2vec.wv
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=padding_index)  # embedding layer

    def forward(self, src, hidden=None):
        embeddings = self.embedding(src)  # (batch_size, max_caption_length, embed_dim)
        embedded = F.dropout(embeddings, p=self.embedding_dropout, inplace=True)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = torch.tanh(self.linear(outputs))
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size, 1))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return torch.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        #energy = energy.transpose(1, 2)  # [B*H*T]
        #v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        #energy = torch.bmm(v, energy)  # [B*1*T]
        energy = (energy @ self.v).squeeze(2)
        return energy.squeeze(1)  # [B*T]

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, padding_index=3, dropout=0.0, embedding_dropout=0.0):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.dropout = dropout
        self.embedding_dropout = embedding_dropout

        self.embed = nn.Embedding(output_size, embed_size, padding_idx=padding_index)
        self.attention = Attention(hidden_size)
        self.gru = SRU(hidden_size + embed_size, hidden_size, num_layers=n_layers,
                       layer_norm=True, dropout=dropout)
        # self.out = nn.Linear(hidden_size*2, output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = F.dropout(self.embed(input), p=self.embedding_dropout, inplace=True).unsqueeze(0)  # (1,B,N)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        if embedded.shape == torch.Size([1, 1, 1, 256]):
            embedded = embedded.squeeze(0)
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        # context = context.squeeze(0)
        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, beam_size, dropout=0.0):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(self.encoder.hidden_size, self.decoder.hidden_size)
        self.dropout = dropout
        self.beam_size = beam_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        beam_width = self.beam_size
        topk = 1
        decoded_batch = list()
        batch_size = 1
        max_len = trg.size(1)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[-self.decoder.n_layers:]
        hidden = torch.tanh(self.encoder.linear(hidden))
        hidden = torch.tanh(self.linear(hidden))
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, batch_size + 1):
            if t != 1:
                v, output = output[0].max(0)
                output = output.unsqueeze(0)
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            values, indices = output[0].max(0)
            decoder_input = torch.LongTensor([[indices]]).cuda()
            endnodes = list()
            number_required = min((topk + 1), topk - len(endnodes))
            node = BeamSearchNode(hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()
            #nodes = Queue()
            nodes.put((-node.eval(), node))
            qsize = 1
            while True:
                # give up when decoding takes too long
                if qsize > 2000: 
                    break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == 2 and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, 
                decoder_hidden, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch