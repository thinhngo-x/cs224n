#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1e
    sanity_check.py 1f
    sanity_check.py 1g
    sanity_check.py 1h
    sanity_check.py 2a
    sanity_check.py 2b
    sanity_check.py 2c
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, batch_iter, read_corpus
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT
from highway import Highway
from cnn import CNN


import torch
import torch.nn as nn
import torch.nn.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 4
DROPOUT_RATE = 0.0

class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_pad = self.char2id['∏']
        self.char_unk = self.char2id['Û']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def reinitialize_layers(model):
    """Reinitialize the layer weights for sainity check
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)

def question_1e_sanity_check():
    """ Sanity check for to_input_tensor_char() function.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1e: To Input Tensor Char")
    print ("-"*80)
    vocabEntry = VocabEntry()

    print("Running test on a list of sentences")
    sentences = [['Human', ':', 'What', 'do', 'we', 'want', '?'], ['Computer', ':', 'Natural', 'language', 'processing', '!'], ['Human', ':', 'When', 'do', 'we', 'want', 'it', '?'], ['Computer', ':', 'When', 'do', 'we', 'want', 'what', '?']]
    sentence_length = 8
    BATCH_SIZE = 4
    word_length = 12
    output = vocabEntry.to_input_tensor_char(sentences, 'cpu')
    output_expected_size = [sentence_length, BATCH_SIZE, word_length]
    assert list(output.size()) == output_expected_size, "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))

    print("Sanity Check Passed for Question 1e: To Input Tensor Char!")
    print("-"*80)

def question_1f_sanity_check():
    """Sanity check for highway.py
    """
    print("-"*80)
    print("Running Sanity Check for Question 1f: Highway")
    print("-"*80)

    torch.manual_seed(0)
    conv_out = torch.zeros(BATCH_SIZE, EMBED_SIZE) + 1
    # print(conv_out)
    highway = Highway(EMBED_SIZE)
    highway.eval()
    reinitialize_layers(highway)
    # for name, param in highway.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    assert highway(conv_out).shape == conv_out.shape, "output shape is incorrect"
    assert np.all(np.fabs(highway.gate(conv_out).detach().numpy() - 0.7311) < 1e-3), "gate value is incorrect"
    assert np.all(np.fabs(highway.proj(conv_out).detach().numpy() - 1) < 1e-3), "projection value is incorrect"
    assert torch.all(torch.eq(highway(conv_out), conv_out)), "output values is incorrect: it should be:\n {} but is:\n{}".format(conv_out, highway(conv_out))

    print("Sanity Check Passed for Question 1f: Highway!")
    print("-"*80)

def question_1g_sanity_check():
    """Sanity check for cnn.py
    """
    print("-"*80)
    print("Running Sanity Check for Question 1g: CNN")
    print("-"*80)

    torch.manual_seed(0)
    length_of_longest_word = 10
    words = torch.randn(BATCH_SIZE, EMBED_SIZE, length_of_longest_word)
    conv = CNN(EMBED_SIZE, HIDDEN_SIZE)
    conv.eval()
    reinitialize_layers(conv)

    assert conv(words).shape == (BATCH_SIZE, HIDDEN_SIZE)

    print("Sanity Check Passed for Question 1g: CNN!")
    print("-"*80)


def question_1h_sanity_check(model):
    """ Sanity check for model_embeddings.py
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1h: Model Embedding")
    print ("-"*80)
    sentence_length = 10
    max_word_length = 21
    inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
    ME_source = model.model_embeddings_source
    output = ME_source.forward(inpt)
    output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    print("Sanity Check Passed for Question 1h: Model Embedding!")
    print("-"*80)

def question_2a_sanity_check(decoder, char_vocab):
    """ Sanity check for CharDecoder.forward()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2a: CharDecoder.forward()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    logits, (dec_hidden1, dec_hidden2) = decoder.forward(inpt)
    logits_expected_size = [sequence_length, BATCH_SIZE, len(char_vocab.char2id)]
    dec_hidden_expected_size = [1, BATCH_SIZE, HIDDEN_SIZE]
    assert(list(logits.size()) == logits_expected_size), "Logits shape is incorrect:\n it should be {} but is:\n{}".format(logits_expected_size, list(logits.size()))
    assert(list(dec_hidden1.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden1.size()))
    assert(list(dec_hidden2.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden2.size()))
    print("Sanity Check Passed for Question 2a: CharDecoder.forward()!")
    print("-"*80)

def question_2b_sanity_check(decoder):
    """ Sanity check for CharDecoder.train_forward()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2b: CharDecoder.train_forward()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    loss = decoder.train_forward(inpt)
    assert(list(loss.size()) == []), "Loss should be a scalar but its shape is: {}".format(list(loss.size()))
    print("Sanity Check Passed for Question 2b: CharDecoder.train_forward()!")
    print("-"*80)

def question_2c_sanity_check(decoder):
    """ Sanity check for CharDecoder.decode_greedy()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2c: CharDecoder.decode_greedy()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)
    initialStates = (inpt, inpt)
    device = decoder.char_output_projection.weight.device
    decodedWords = decoder.decode_greedy(initialStates, device)
    assert(len(decodedWords) == BATCH_SIZE), "Length of decodedWords should be {} but is: {}".format(BATCH_SIZE, len(decodedWords))
    print("Sanity Check Passed for Question 2c: CharDecoder.decode_greedy()!")
    print("-"*80)

def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = NMT(
        word_embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    if args['1e']:
        question_1e_sanity_check()
    elif args['1h']:
        question_1h_sanity_check(model)
    elif args['2a']:
        question_2a_sanity_check(decoder, char_vocab)
    elif args['2b']:
        question_2b_sanity_check(decoder)
    elif args['2c']:
        question_2c_sanity_check(decoder)
    elif args['1f']:
        question_1f_sanity_check()
    elif args['1g']:
        question_1g_sanity_check()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
