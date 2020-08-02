#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, embed_size, dropout_rate=0.3):
    	""" Init highway network.

		@param embed_size (int): Embedding size (dimensionality) of word
		@param dropout_rate (float): Dropout probability, for output

    	"""
    	super(Highway, self).__init__()
    	def init_weights(m):
	        if type(m) == nn.Linear:
	            if m.bias is not None:
	                nn.init.constant_(m.bias, -1)

    	self.embed_size = embed_size
    	self.gate = nn.Sequential(
    		nn.Linear(embed_size, 1),
    		nn.Sigmoid()
    		)
    	self.gate.apply(init_weights)
    	self.proj = nn.Sequential(
    		nn.Linear(embed_size, embed_size),
    		nn.ReLU()
    		)
    	self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x_conv_out):
    	""" Takes a mini batch of x_conv_out to compute the output fo the highway layer.

    	@param x_conv_out (Tensor): A tensor of shape (batch_size, embed_size)
		@returns x_highway (Tensor): A tensor of shape (batch_size, embed_size)

    	"""
    	gate = self.gate(x_conv_out)
    	proj = self.proj(x_conv_out)
    	x_highway = gate * proj + (1-gate) * x_conv_out

    	return self.dropout(x_highway)

    ### END YOUR CODE

