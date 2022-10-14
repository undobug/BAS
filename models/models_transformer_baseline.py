# --------------------------------------------------------
#!/usr/bin/env python
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np
import math


class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def masked_softmax(X, valid_lens):

    return nn.functional.softmax(X, dim=-1)


def transpose_qkv(X, num_heads):

    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):

    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        scores = scores 
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        attention = torch.bmm(self.dropout(self.attention_weights), values)
        return attention


class EncoderBlock(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))


class attBlock(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(attBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return Y
 

class sffnBlock(nn.Module):

    def __init__(self, num_hiddens, ffn_num_input, ffn_num_hiddens, **kwargs):
        super(sffnBlock, self).__init__(**kwargs)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)

    def forward(self, X):
        Y = self.ffn(X)
        return Y


class MultiHeadAttention(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, queries, keys, values):

        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class TransformerEncoder(nn.Module):

    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, X,valid_lens, *args):

        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X,valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class MixedOperation(nn.Module):
    
    def __init__(self,EncoderBlock, attBlock, sffnBlock):
        super(MixedOperation, self).__init__()

        self.ops = nn.ModuleList([EncoderBlock, attBlock, sffnBlock])
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(self.ops) for i in range(len(self.ops))]))
    
    def forward(self, x, temperature=1):
        if self.training == True:
            soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        if self.training == False:
            soft_mask_variables = F.one_hot(self.thetas.argmax(),3)

        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))

        return output,soft_mask_variables


class SER_search(nn.Module):
    
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, num_classes, use_bias=False):
        super(SER_search, self).__init__()
        self.num_hiddens = num_hiddens
        self.stages_to_search = nn.ModuleList([MixedOperation(
                                                   EncoderBlock(key_size, query_size, value_size, num_hiddens,norm_shape, 
                                                   ffn_num_input, ffn_num_hiddens, num_heads,dropout, use_bias=use_bias),
                                                   attBlock(key_size, query_size, value_size, num_hiddens,
                                                             norm_shape, num_heads,dropout, use_bias=use_bias),
                                                   sffnBlock(num_hiddens, ffn_num_input, ffn_num_hiddens))
                                               for i in range(num_layers)])
        
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(self.num_hiddens, self.num_hiddens // 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.num_hiddens // 2, self.num_hiddens // 4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.num_hiddens // 4, num_classes),
        )

    def forward(self, x):
        valid_lens =None
        inx = []
        for mixed_op in self.stages_to_search:
            x,weights = mixed_op(x)
            index = torch.argmax(weights)
            inx.append(index)
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze(dim=2)
        x = self.classifier(x)
        return x