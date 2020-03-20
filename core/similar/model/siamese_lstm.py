#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SiameseLstm(nn.Module):
    def __init__(self, embed_dim, hidden_dim, device, maxwords=6000):
        super(SiameseLstm, self).__init__()
        self.device = device
        self.embedding_layer = nn.Embedding(maxwords, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)


    def forward_once(self, q, q_len):
        sorted_indices = torch.argsort(q_len, descending=True)
        _q = torch.index_select(q, dim=0, index=sorted_indices)
        q_len = torch.index_select(q_len, dim=0, index=sorted_indices)
        embedding = self.embedding_layer(_q)
        packed = pack_padded_sequence(embedding, q_len, batch_first=True)
        out, (_, _) = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(out, batch_first=True)
        result = torch.zeros(unpacked.size())
        for i, encoded_matrix in enumerate(unpacked):
            result[sorted_indices[i]] = encoded_matrix
        return result

    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=0)).to(self.device)

    def forward(self, x1, x1_len, x2, x2_len):
        out_1 = self.forward_once(x1, x1_len)
        out_2 = self.forward_once(x2, x2_len)
        similarity_score = torch.zeros(out_1.size()[0]).to(self.device)
        for index in range(out_1.size()[0]):
            q1 = out_1[index, x1_len[index] - 1, :]
            q2 = out_2[index, x2_len[index] - 1, :]
            similarity_score[index] = self.exponent_neg_manhattan_distance(q1, q2)
        # q1_emb = self.embedding_layer(x1)
        # q2_emb = self.embedding_layer(x2)
        # q1_out, _ = self.lstm(q1_emb, None)
        # q2_out, _ = self.lstm(q2_emb, None)
        # manhattan_dis = torch.exp(-torch.sum(torch.abs(out_1[:, -1, :] - out_2[:, -1, :]), dim=1, keepdim=True))
        # return manhattan_dis
        return similarity_score
