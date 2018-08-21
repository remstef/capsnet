# -*- coding: utf-8 -*-

import torch

class ReClass(torch.nn.Module):
  ''' `convtest`
    seqlen = 5
    nfeat = 4
    x = torch.Tensor(list(range(2*seqlen*nfeat))).view(2,1,seqlen,nfeat) # batch x channnel x seqlen x features
    numfilters=3
    convwindow=2
    conv = torch.nn.Conv2d(1, numfilters, (convwindow, nfeat))
    relu = torch.nn.ReLU()
    maxpool = torch.nn.MaxPool2d((seqlen - (convwindow-1), 1) )
    y = conv(x)
    y_ = relu(y)
    y__ = maxpool(y_)
  '''
  def __init__(self, 
               ntoken,
               nclasses,
               maxseqlength,
               maxentlength,
               maxdist,
               window_size,
               emsizeword,
               emsizeposi,
               emsizeclass,
               nhid,
               numconvfilters=100,
               convwindow=1,
               dropout=0.2,
               conv_activation='ReLU',
               weightsword=None,
               fix_emword=False):
    
    super(ReClass, self).__init__()
    
    self.window_size = window_size
    self.fs = window_size * (emsizeword + 2 * emsizeposi) # size of the feature vector for words
    
    # layers
    self.word_embeddings = torch.nn.Embedding(ntoken, emsizeword)
    self.posi_embeddings = torch.nn.Embedding(maxdist * 2 + 1, emsizeposi)
    self.class_embeddings = torch.nn.Embedding(nclasses, emsizeclass)
    self.d1 = torch.nn.Dropout(dropout)
    
    self.conv = torch.nn.Conv2d(1, numconvfilters, (convwindow, self.fs), bias=True) #bias??
    
    if not conv_activation in ['ReLU', 'Tanh']:
      raise ValueError( '''Invalid option `%s` for 'conv-activation'.''' % conv_activation)
    self.convact = getattr(torch.nn, conv_activation)()
    self.d2 = torch.nn.Dropout(dropout)
    self.maxpool = torch.nn.MaxPool2d(((maxseqlength-window_size//2-1) - (convwindow-1),1))
    self.linear = torch.nn.Linear(numconvfilters, nhid)

    self.linear2 = torch.nn.Linear((maxentlength * emsizeword * 2) + nhid, nclasses)
    self.d3 = torch.nn.Dropout(dropout)
    self.softmax = torch.nn.LogSoftmax(dim=1) # Softmax(dim=1)

    # initialization actions
    self.init_weights(weightsword, fix_emword)
    
    
  def init_weights(self, weightsword, fix_emword):
    initrange = 0.1
    # TODO: check if that makes any difference
    #self.posi_embeddings.bias.data.zero_()
    #self.word_embeddings.bias.data.zero_()
    #self.class_embeddings.bias.data.zero_()    
    if weightsword is None:
      self.word_embeddings.weight.data.uniform_(-initrange, initrange)
    else:
      assert weightsword.size() == self.word_embeddings.weight.size(), f'Size clash emwords supplied weights: {weightsword.size()}, expected {self.word_embeddings.weight.size()}'
      self.word_embeddings.load_state_dict({'weight': weightsword})
    if fix_emword:
      self.word_embeddings.weight.requires_grad = False
    self.posi_embeddings.weight.data.uniform_(-initrange, initrange)
    self.class_embeddings.weight.data.uniform_(-initrange, initrange)

  def forward(self, seq, seqlen, e1, e1len, e2, e2len, offs_e1, offs_e2, seqp_e1, seqp_e2):
    # seq = batch_size x max_seq_length (padded) : sentence
    # seqlen = batch_size x seq_length
    # e1 & e2 = batch_size x 2 : offsets as begin e[0] and end e[1]
    # seqpX = batch_size x max_seqlength (padded) : sentence as relative position indices to e1 and e2
    
    ## BEGIN: sentence level features
    we = self.word_embeddings(seq)
    pe1 = self.posi_embeddings(seqp_e1)
    pe2 = self.posi_embeddings(seqp_e2)
    
    # concatenate word embedding with positional embedding, w = batch_size x seq_length x (wemsize+2xpemsize)
    w = torch.cat((we, pe1, pe2), dim=2)
    w = self.d1(w) # because its a good poilicy
    # concatenate embeddings their context embeddings in a sliding window fashion, w = batch_size x  seq_length-windowsize//2-1 x (windowsize x (wemsize+2xpemsize))
    w = self.window_cat(w, self.window_size)
    
    # convolution + maxpooling
    w.unsqueeze_(1) # add `channel` dimension; needed for conv: w = batch_size x 1 x seq_length x nfeatures
    c = self.conv(w)
    c = self.convact(c) # because it's a good policy
    c = self.d2(c) # yet another good policy, although debatable if it should come here
    f = self.maxpool(c)
    f.squeeze_() # remove trailing singular dimensions (f: batch_size x numfilters x 1 x 1 => batch_size x numfilters)
    
    # linear classification
    o1 = self.linear(f)
    ## END: sentence level features
    
    ## BEGIN: lexical level features
    L1 = self.word_embeddings(e1)
    L1 = L1.view(L1.size(0), -1) # concatenate entity embedding vectors
    L2 = self.word_embeddings(e2)
    L2 = L2.view(L2.size(0), -1) # concatenate entity embedding vectors, keep batch dimension (0) in tact
    l = torch.cat((L1, L2, o1), dim=1) # concatenate features vectors
    l = self.d3(l)
    ## END: lexical level features

    o2 = self.linear2(l)
    o = self.softmax(o2)
    
    return o, 0
  
  @staticmethod
  def window_cat(seq, n):
    '''
    in:  seq = batch_size x seqence x features
    out:       batch_size x seqence-n//2-1 x (features x n)
    
    concatenate sliding windows (proper padding of sequences beforehand is expected)

    x = torch.Tensor(list(range(3*4*5))).view(3,4,5)
    tensor([[[ 0.,  1.,  2.,  3.,  4.],
             [ 5.,  6.,  7.,  8.,  9.],
             [10., 11., 12., 13., 14.],
             [15., 16., 17., 18., 19.]],

            [[20., 21., 22., 23., 24.],
             [25., 26., 27., 28., 29.],
             [30., 31., 32., 33., 34.],
             [35., 36., 37., 38., 39.]],

            [[40., 41., 42., 43., 44.],
             [45., 46., 47., 48., 49.],
             [50., 51., 52., 53., 54.],
             [55., 56., 57., 58., 59.]]])

    y = ReClass.window_cat(x, 3)
    tensor([[[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
             [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.]],
 
            [[20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.],
             [25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39.]],

            [[40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54.],
             [45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.]]])
    '''
    return torch.cat([seq[:,i:seq.size(1)-(n-i-1),:] for i in range(n)],dim=2)
  
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
      h = (w.new_zeros(self.nlayers, bsz, self.nhid),
           w.new_zeros(self.nlayers, bsz, self.nhid))
    else:
      h = w.new_zeros(self.nlayers, bsz, self.nhid)   
    return h
    
  def repackage_hidden(self, h):
    '''Wraps hidden states in new Tensors, to detach them from their history.'''
    if isinstance(h, torch.Tensor):
      return h.detach()
    else:
      return tuple(self.repackage_hidden(v) for v in h)
    
  def sort_padded_inputs_by_length(self, x, lengths):
    lengths_sorted, idx = lengths.sort(dim=0, descending=True)
    _, invidx = idx.sort() # prepare inverted index in order to restore original ordering later
    y = x[:lengths_sorted[0],idx,...] # reorder and trim to longest sequence in the batch
    return y, lengths_sorted, idx, invidx

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

class RNN_LM_simple(torch.nn.Module):
  '''
  
  '''
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


class RNN_CLASSIFY_simple(torch.nn.Module):
  '''
  inspired by https://discuss.pytorch.org/t/lstm-for-many-to-one-multiclass-classification-problem/14268/5
  '''
  def __init__(self, ntoken, nhid, nlayers, nclasses, *args, **kwargs):
    super(RNN_CLASSIFY_simple, self).__init__()
    self.ntoken = ntoken
    self.nhid = nhid
    self.nlayers = nlayers
    self.lstm = torch.nn.LSTM(ntoken, nhid, nlayers, batch_first=True)
    self.fc = torch.nn.Linear(nhid, nclasses)
  
  def forward(self, x):
    # Set initial hidden and cell states 
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device()) 
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device())
    
    # Forward propagate LSTM
    out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
    
    # Decode the hidden state of the last time step
    out = self.fc(out[:, -1, :])
    return out

  
class RNN_CLASSIFY_linear(torch.nn.Module):
  '''
   taken from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
  '''    
  def __init__(self, ntoken, nhid, nclasses, *args, **kwargs):
    super(RNN_CLASSIFY_linear, self).__init__()
    self.ntoken = ntoken
    self.nhid = nhid
    self.nclasses = nclasses
    self.i2h = torch.nn.Linear(ntoken + nhid, nhid)
    self.i2o = torch.nn.Linear(ntoken + nhid, nclasses)
    self.softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, inputs, hidden):
    # inputs = batch_size x ntoken
    i_h = torch.cat((inputs, hidden), dim=1)
    h = self.i2h(i_h)
    o = self.i2o(i_h)
    o = self.softmax(o)
    return o, h

  def init_hidden(self, batch_size):
    w = next(self.parameters())
    return w.new_zeros(batch_size, self.nhid).uniform_(-.1, .1)
  
  



