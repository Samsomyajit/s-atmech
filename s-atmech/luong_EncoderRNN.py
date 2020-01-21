"""
This is the Encoder RNN layer for implementing the Luong attention courtesy:https://arxiv.org/pdf/1508.04025.pdf
the code is made by team Bread and Code's Somyajit Chakraborty for s-atmech
"""
import torch 
import logging
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class luong_EncoderRNN(nn.Module):
  def __init__(self , hidden_size , input_size , n_layers , dropout , word_embedding_matrix , rnn_cell , use_cuda)
      super().__init__()
      self.hidden_size = hidden_size
      self.input_size = input_size
      self.n_layers = n_layers
      self.dropout = dropout
      self.word_embedding_matrix = word_embedding_matrix
      self.rnn_cell = rnn_cell
      self.use_cuda = use_cuda
      
      if rnn.cell == "GRU"
          self.rnn = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
          
      def forward(self, input_seqs, hidden, input_lengths)
            """
              input_seqs : (Max input length, batch_size)
              input_lengths: (batch_size)
            """
      
            
            embedded = self.embedding(input_seqs)
            packed = pack_padded_sequence(embedded, input_lengths)
            outputs, hidden = self.rnn(packed, hidden)
            outputs, output_lengths = pad_packed_sequence(outputs)
            
            # Max input_lenghts, batch_size, hidden_size, we add backward and forward
            # hidden states together
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            
            # get the forward and backward hidden states
            
            hidden_layers = []
            for i in range(self.n_layers):
                    hidden_layers.append((hidden[i * 2, :, :] + hidden[(i * 2) + 1, :, :]).unsqueeze(0))
            return outputs, torch.cat(hidden_layers, 0)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
        if self.use_cuda: hidden = hidden.cuda()
        return hidden
