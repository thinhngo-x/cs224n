#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, embed_size_char, embed_size_word):
    	"""Init convolutional neural network

    	@param embed_size_char (int): Embedding size of characters
    	@param embed_size_word (int): Embedding size of words

    	"""
    	super(CNN, self).__init__()
    	self.convo = nn.Conv1d(embed_size_char, embed_size_word, 5, padding=1)

    def forward(self, x):
    	"""Takes a minibatch of embedding characters of words to compute their embedding vectors

		@param x (Tensor): A tensor of shape (batch_size, embed_size_char, length_of_longest_word)

		@returns conv_out (Tensor): A tensor of shape (batch_size, embed_size_word)

    	"""
    	x = F.relu(self.convo(x))
    	conv_out = F.max_pool1d(x, x.shape[-1]).squeeze(-1)
    	return conv_out

    ### END YOUR CODE

