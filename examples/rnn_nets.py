# -*- coding: utf-8 -*-

import torch


class RNNLM(torch.nn.Module):
  '''https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967'''

  def __init__(
      self, 
      rnn_type, 
      ntoken, 
      ninp, # = emsize
      nhid, 
      nlayers, 
      dropout=0.5, 
      tie_weights=False, 
      init_em_weights=None, 
      train_em_weights=True):
    
    super(RNNLM, self).__init__()
    self.drop = torch.nn.Dropout(dropout)
    self.encoder = torch.nn.Embedding(ntoken, ninp)
    if rnn_type in ['LSTM', 'GRU']:
      self.rnn = getattr(torch.nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
    else:
      try:
        nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
      except KeyError: raise ValueError( '''An invalid option `%s` for 'rnntype' was supplied, \noptions are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']''' % rnn_type)
      self.rnn = torch.nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
    self.decoder = torch.nn.Linear(nhid, ntoken)
    self.init_weights(init_em_weights, train_em_weights)
    if tie_weights:
      if nhid != ninp: raise ValueError('When using the tied flag, nhid must be equal to emsize')
      self.decoder.weight = self.encoder.weight
    self.rnn_type = rnn_type
    self.nhid = nhid
    self.nlayers = nlayers

  def init_weights(self, w = None, trainable = True):
    initrange = 0.1
    if w is None:
      self.encoder.weight.data.uniform_(-initrange, initrange)
    else:
      assert w.size() == self.encoder.weight.size()
      self.encoder.load_state_dict({'weight': w})
      if not trainable:
        self.encoder.weight.requires_grad = False
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def init_hidden(self, bsz):
    w = next(self.parameters())
    if self.rnn_type == 'LSTM':
      return (w.new_zeros(self.nlayers, bsz, self.nhid),
              w.new_zeros(self.nlayers, bsz, self.nhid))
    else:
      return w.new_zeros(self.nlayers, bsz, self.nhid)

  def forward(self, inputs, hidden, seqlengths = None):
    # inputs.size() should be = seq_len, batch_size, feature_size (1 = word index)
    e = self.encoder(inputs)
    e = self.drop(e)
    pack_sequences = seqlengths is not None and len(seqlengths.unique()) > 1 # if all sequences have the same lengths, they don't need to be packed     
    if pack_sequences: # sequences are padded and must be ordered by seqlength in order to pack them
      # 1. sort sequences by length; 2. create a PackedSequence from the padded sequences
      e, seqlengths_sorted, _, invidx = self.sort_padded_inputs_by_length(e, seqlengths)
      e = torch.nn.utils.rnn.pack_padded_sequence(e, seqlengths_sorted, batch_first = False) # unpad
    o, h = self.rnn(e, hidden)
    if pack_sequences: 
      # 1. unpack PackedSequence to padded sequence; 2. restore original ordering
      o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first = False, total_length = inputs.size(0)) # pad again
      o = o[:,invidx,...]
    o = self.drop(o)
    d = self.decoder(o.view(o.size(0)*o.size(1), o.size(2)))
    d = d.view(o.size(0), o.size(1), d.size(1))
    return d, h

  def sort_padded_inputs_by_length(self, x, lengths):
    lengths_sorted, idx = lengths.sort(dim=0, descending=True)
    _, invidx = idx.sort() # prepare inverted index in order to restore original ordering later
    y = x[:lengths_sorted[0],idx,...] # reorder and trim to longest sequence in the batch
    return y, lengths_sorted, idx, invidx

'''
taken from https://github.com/pytorch/examples/tree/master/word_language_model
'''      
class RNN_LM_original(torch.nn.Module):
  """Container module with an encoder, a recurrent module, and a decoder."""
  '''https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967'''

  def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, **ignoredkwargs):
    print('Ignored args: %s'  % ignoredkwargs)
    super(RNN_LM_original, self).__init__()
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

    self.init_weights()
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

  def init_weights(self):
    initrange = 0.1
    self.encoder.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)

  def forward(self, inputs, hidden):
    # inputs.size() should be = seq_len, batch_size, feature_size (1 = word index)
    e = self.encoder(inputs)
    e = self.drop(e)
    o, h = self.rnn(e, hidden)
    o = self.drop(o)
    d = self.decoder(o.view(o.size(0)*o.size(1), o.size(2)))
    d = d.view(o.size(0), o.size(1), d.size(1))
    return d, h

  def init_hidden(self, bsz):
    w = next(self.parameters())
    if self.rnn_type == 'LSTM':
      return (w.new_zeros(self.nlayers, bsz, self.nhid),
              w.new_zeros(self.nlayers, bsz, self.nhid))
    else:
      return w.new_zeros(self.nlayers, bsz, self.nhid)

'''

'''
class RNN_LM_simple(torch.nn.Module):
  def __init__(self, ntoken, emsize, nhid, nlayers=1):
    super(RNN_LM_simple, self).__init__()
    self.ntoken = ntoken
    self.emsize = emsize
    self.nhid = nhid
    self.nlayers = nlayers

    self.encoder = torch.nn.Embedding(ntoken, emsize)
    self.rnn = torch.nn.GRU(emsize, nhid, nlayers)
    self.decoder = torch.nn.Linear(nhid, ntoken)

  def forward(self, inputs, hidden):
    e = self.encoder(inputs.view(1, -1))
    o, h = self.rnn(e.view(1, 1, -1), hidden)
    o = self.decoder(o.view(1, -1))
    return o, h

  def init_hidden(self):
    return torch.autograd.Variable(torch.zeros(self.nlayers, 1, self.nhid))
  
'''
 taken from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
'''  
class RNN_CLASSIFY_simple(torch.nn.Module):
  
  def __init__(self, ntokens, nhid, nclasses):
    super(RNN_CLASSIFY_simple, self).__init__()
    self.nhid = nhid
    self.i2h = torch.nn.Linear(ntokens + nhid, nhid)
    self.i2o = torch.nn.Linear(ntokens + nhid, nclasses)
    self.softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, inputs, hidden):
    i_h = torch.cat((inputs, hidden), dim=1)
    h = self.i2h(i_h)
    o = self.i2o(i_h)
    o = self.softmax(o)
    return o, h

  def initHidden(self):
    w = next(self.parameters())
    return w.new_zeros(1, self.nhid)
#    return torch.zeros(1, self.hidden_size)
  
  



