# Import Module
import os
import re
import time
import glob
import pickle
import gensim
import argparse
import warnings
import sentencepiece as spm
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from PyRouge.pyrouge import Rouge

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# Import Custom Module
from module_beam import Encoder, Decoder, Seq2Seq
from utils import getDataLoader, CustomDataset
from train_utils import test

def main(args):
    # Setting
    warnings.simplefilter("ignore", UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Args Parser
    hj_method = args.hj_method
    kr_method = args.kr_method
    batch_size = args.batch_size
    beam_size = args.beam_size
    hidden_size = args.hidden_size
    embed_size = args.embed_size
    vocab_size = args.vocab_size
    max_len = args.max_len
    padding_index = args.pad_id
    n_layers = args.n_layers
    stop_ix = args.stop_ix

    # Load saved model & Word2vec
    save_path = 'save_{}_{}_{}_maxlen_{}'.format(vocab_size, hj_method, kr_method, max_len)
    save_list = sorted(glob.glob(f'./save/{save_path}/*.*'))
    save_pt = save_list[-1]
    print('Will load {} pt file...'.format(save_pt))
    word2vec_hj = Word2Vec.load('./w2v/word2vec_hj_{}_{}.model'.format(vocab_size, hj_method))

    # SentencePiece model load
    spm_kr = spm.SentencePieceProcessor()
    spm_kr.Load("./spm/m_korean_{}.model".format(vocab_size))

    # Test data load
    with open('./test_dat.pkl', 'rb') as f:
        test_dat = pickle.load(f)

    test_dataset = CustomDataset(test_dat['test_hanja'], test_dat['test_korean'])
    test_loader = getDataLoader(test_dataset, pad_index=padding_index, shuffle=False, batch_size=batch_size)

    # Model load
    print('Model loading...')
    encoder = Encoder(vocab_size, embed_size, hidden_size, word2vec_hj, n_layers=n_layers, padding_index=padding_index)
    decoder = Decoder(embed_size, hidden_size, vocab_size, n_layers=n_layers, padding_index=padding_index)
    seq2seq = Seq2Seq(encoder, decoder, beam_size).cuda()
    #optimizer = optim.Adam(seq2seq.parameters(), lr=lr, weight_decay=w_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=lr_decay)
    print(seq2seq)

    print('Testing...')
    start_time = time.time()
    results = test(seq2seq, test_loader, vocab_size, load_pt=save_pt, stop_ix=stop_ix)
    print(time.time() - start_time)
    print('Done!')

    print("Decoding...")
    pred_list = list()
    for result_text in tqdm(results):
        text = torch.Tensor(result_text).squeeze().tolist()
        text = [int(x) for x in text]
        prediction_sentence = spm_kr.decode_ids(text).strip() # Decode with strip
        pred_list.append(prediction_sentence)
    ref_list = list()
    for ref_text in tqdm(test_dat['test_korean'][:stop_ix]):
        ref_list.append(spm_kr.decode_ids(ref_text).strip())
    print('Done!')

    with open(f'./save/{save_path}/test_result.pkl', 'wb') as f:
        pickle.dump({
            'pred': pred_list,
            'reference': ref_list,
        }, f)
    print('Save file; /test_dat.pkl')

    # Calculate BLEU Score
    print('Calculate BLEU4, METEOR, Rogue-L...')
    chencherry = SmoothingFunction()
    bleu4 = corpus_bleu(test_dat['reference'], test_dat['pred'], 
                        smoothing_function=chencherry.method4)
    print('BLEU Score is {}'.format(bleu4))

    # Calculate METEOR Score
    meteor = meteor_score(test_dat['reference'], test_dat['pred'])
    print('METEOR Score is {}'.format(meteor))

    # Calculate Rouge-L Score
    r = Rouge()
    total_test_length = len(test_dat['reference'])
    precision_all = 0
    recall_all = 0
    f_score_all = 0
    for i in range(total_test_length):
        [precision, recall, f_score] = r.rouge_l([test_dat['reference'][i]], [test_dat['pred'][i]])
        precision_all += precision
        recall_all += recall
        f_score_all += f_score
    print('Precision : {}'.foramt(round(precision_all/total_test_length, 4)))
    print('Recall : {}'.foramt(round(recall_all/total_test_length, 4)))
    print('F Score : {}'.foramt(round(f_score_all/total_test_length, 4)))


if __name__ == '__main__':
    # Args Parser
    parser = argparse.ArgumentParser(description='Joseon NMT argparser')
    parser.add_argument('--vocab_size', type=int, default=30000,
                        help='Preprocessed vocabulary size; Default is 30000')
    parser.add_argument('--hj_method', type=str, default='spm', 
                        help='Hanja parsing method: spm / chr / chr_cnn; Default is spm')
    parser.add_argument('--kr_method', type=str, default='spm', 
                        help='Korean parsing method: spm / khaiii; Default is spm')
    parser.add_argument('--beam_size', type=int, default=5, 
                        help='Beam size; Default is 5')
    parser.add_argument('--pad_id', default=0, type=int, help='pad index')
    parser.add_argument('--num_epoch', type=int, default=10, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size; Default is 1')
    parser.add_argument('--n_layers', type=int, default=5, help='Model layers; Default is 5')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden State Vector Dimension; Default is 256')
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding Vector Dimension; Default is 256')
    parser.add_argument('--max_len', type=int, default=150, help='Max Length of Sentence; Default is 150')
    parser.add_argument('--stop_ix', type=int, default=None, 
                        help='Stopping index for testing. If you wanna sample the results, check this; Default is None')
    args = parser.parse_args()

    main(args)