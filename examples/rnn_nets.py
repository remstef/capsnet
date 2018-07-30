# -*- coding: utf-8 -*-

import torch

class RNNLM(torch.nn.Module):
  """Container module with an encoder, a recurrent module, and a decoder."""
  '''https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967'''

  def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, init_em_weights=None, train_em_weights=True):
    super(RNNLM, self).__init__()
    self.drop = torch.nn.Dropout(dropout)
    self.encoder = torch.nn.Embedding(ntoken, ninp)
    if rnn_type in ['LSTM', 'GRU']:
      self.rnn = getattr(torch.nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
    else:
      try:
        nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
      except KeyError:
        raise ValueError( '''An invalid option `%s` for 'rnntype' was supplied,
                           options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']''' % rnn_type)
      self.rnn = torch.nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
    self.decoder = torch.nn.Linear(nhid, ntoken)

    self.init_weights(init_em_weights, train_em_weights)
    # Optionally tie weights as in:
    # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
    # https://arxiv.org/abs/1608.05859
    # and
    # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
    # https://arxiv.org/abs/1611.01462
    if tie_weights:
      if nhid != ninp:
        raise ValueError('When using the tied flag, nhid must be equal to emsize')
      self.decoder.weight = self.encoder.weight

    self.rnn_type = rnn_type
    self.nhid = nhid
    self.nlayers = nlayers

  def init_weights(self, weights = None, trainable = True):
    initrange = 0.1
    if weights is None:
      self.encoder.weight.data.uniform_(-initrange, initrange)
    else:
      assert weights.size() == self.encoder.weight.size()
      self.encoder.load_state_dict({'weight': weights})
      if not trainable:
        self.encoder.weight.requires_grad = False
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, input, hidden):
    emb = self.drop(self.encoder(input))
    output, hidden = self.rnn(emb, hidden)
    output = self.drop(output)
    decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
    return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

  def init_hidden(self, bsz):
    weight = next(self.parameters())
    if self.rnn_type == 'LSTM':
      return (weight.new_zeros(self.nlayers, bsz, self.nhid),
              weight.new_zeros(self.nlayers, bsz, self.nhid))
    else:
      return weight.new_zeros(self.nlayers, bsz, self.nhid)

'''

'''
class SimpleRNN(torch.nn.Module):
  def __init__(self, ntoken, emsize, nhid, nlayers=1):
    super(SimpleRNN, self).__init__()
    self.ntoken = ntoken
    self.emsize = emsize
    self.nhid = nhid
    self.nlayers = nlayers

    self.encoder = torch.nn.Embedding(ntoken, emsize)
    self.gru = torch.nn.GRU(emsize, nhid, nlayers)
    self.decoder = torch.nn.Linear(nhid, ntoken)

  def forward(self, input, hidden):
    input = self.encoder(input.view(1, -1))
    output, hidden = self.gru(input.view(1, 1, -1), hidden)
    output = self.decoder(output.view(1, -1))
    return output, hidden

  def init_hidden(self):
    return torch.autograd.Variable(torch.zeros(self.nlayers, 1, self.nhid))
