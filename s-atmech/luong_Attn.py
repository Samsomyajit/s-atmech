"""
This module implements the Luong Attention Mechanism.
by: Somyajit Chakraborty
"""
import torch
import logging
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class luong_Attn(nn.Module):
    def __init__(self, method, hidden_size, use_cuda=False):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        """
        hidden : 1, batch_size, hidden_size
        encoder_outputs : max input length, batch_size, hidden_size
        """
        # max_len = encoder_outputs.size(0)
        # this_batch_size = encoder_outputs.size(1)

        attn_energies = torch.bmm(self.attn(hidden).transpose(1, 0), encoder_outputs.permute(1, 2, 0))

        # Batch size, 1, max input length
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):

        if self.method == 'general':
            energy = self.attn(encoder_output).view(-1)
            energy = hidden.view(-1).dot(energy)
            return energy
