# -*- coding: utf-8 -*-

# coding: utf-8

import sys
if not '..' in sys.path: sys.path.append('..')

import argparse
import time
import math
import os
import torch
import torchnet
from tqdm import tqdm

from data import WikiSentences
from utils import Index, RandomBatchSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from embedding import Embedding, FastTextEmbedding, TextEmbedding, RandomEmbedding

from rnn_nets import RNNLM

try:

  parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
  parser.add_argument('--data', type=str, default='../data/wikisentences',
                      help='location of the data corpus')
  parser.add_argument('--model', type=str, default='LSTM',
                      help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
  parser.add_argument('--emsize', type=int, default=200,
                      help='size of word embeddings')
  parser.add_argument('--nhid', type=int, default=200,
                      help='number of hidden units per layer')
  parser.add_argument('--nlayers', type=int, default=2,
                      help='number of layers')
  parser.add_argument('--lr', type=float, default=20,
                      help='initial learning rate')
  parser.add_argument('--lr_decay', type=float, default=0.25,
                      help='decay learining learning rate if no validation improvement occurs')
  parser.add_argument('--clip', type=float, default=0.25,
                      help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=40,
                      help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                      help='batch size') 
  parser.add_argument('--bptt', type=int, default=35,
                      help='sequence length') 
  parser.add_argument('--dropout', type=float, default=0.2,
                      help='dropout applied to layers (0 = no dropout)')
  parser.add_argument('--tied', action='store_true',
                      help='tie the word embedding and softmax weights')
  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  parser.add_argument('--cuda', action='store_true',
                      help='use CUDA')
  parser.add_argument('--shuffle_batches', action='store_true',
                      help='shuffle batches')
  parser.add_argument('--shuffle_samples', action='store_true',
                      help='shuffle samples')
  parser.add_argument('--save', type=str, default='model.pt',
                      help='path to save the final model')
  parser.add_argument('--init_weights', type=str, default='',
                      help='path to initial embedding. emsize must match size of embedding')
  args = parser.parse_args()
  
  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
      if not args.cuda:
          print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  
  device = torch.device("cuda" if args.cuda else "cpu")
  
  ###############################################################################
  # Load data
  ###############################################################################
  
  index = Index()
  train_ = WikiSentences(args.data, subset='train', index = index, seqlen = args.bptt).to(device)
  test_ = WikiSentences(args.data, subset='test', index = index, seqlen = args.bptt).to(device)
  valid_ = WikiSentences(args.data, subset='valid', index = index, seqlen = args.bptt).to(device)
  index.freeze().tofile(os.path.join(args.data, 'vocab.txt'))
  
  # load pre embedding
  if args.init_weights:
    # determine type of embedding by checking it's suffix
    if args.init_weights.endswith('bin'):
      preemb = FastTextEmbedding(args.init_weights, normalize = True).load()
      if args.emsize != preemb.dim():
        raise ValueError('emsize must match embedding size. Expected %d but got %d)' % (args.emsize, preemb.dim()))
    elif args.init_weights.endswith('txt'):
      preemb = TextEmbedding(args.init_weights, vectordim = args.emsize).load(normalize = True)
    elif args.init_weights.endswith('rand'):
      preemb = RandomEmbedding(vectordim = args.emsize)
    else:
      raise ValueError('Type of embedding cannot be inferred.')
    preemb = Embedding.filteredEmbedding(index.vocabulary(), preemb, fillmissing = True)
    preemb_weights = torch.Tensor(preemb.weights)
  else:
    preemb_weights = None
  
  eval_batch_size = 10
  
  __ItemSampler = RandomSampler if args.shuffle_samples else SequentialSampler
  __BatchSampler = RandomBatchSampler if args.shuffle_batches else BatchSampler
  
  train_loader = torch.utils.data.DataLoader(train_, batch_sampler = __BatchSampler(__ItemSampler(train_), batch_size=args.batch_size, drop_last = True), num_workers = 0)
  test_loader = torch.utils.data.DataLoader(test_, batch_sampler = __BatchSampler(__ItemSampler(test_), batch_size=eval_batch_size, drop_last = True), num_workers = 0)
  valid_loader = torch.utils.data.DataLoader(valid_, batch_sampler = __BatchSampler(__ItemSampler(valid_), batch_size=eval_batch_size, drop_last = True), num_workers = 0)
  
  ###############################################################################
  # Build the model
  ###############################################################################
  ntokens = len(index)
  model = RNNLM(
      rnn_type = args.model, 
      ntoken = ntokens, 
      ninp = args.emsize, 
      nhid = args.nhid, 
      nlayers = args.nlayers, 
      dropout = args.dropout, 
      tie_weights = args.tied, 
      init_em_weights = preemb_weights, 
      train_em_weights = True).to(device)
  
  criterion = torch.nn.CrossEntropyLoss()
  hidden = None
  
  ###############################################################################
  # Set up Engine
  ###############################################################################
  
  def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
      return h.detach()
    else:
      return tuple(repackage_hidden(v) for v in h)
  
  def process(batch_data):
    global hidden, lr
    
    x_batch, y_batch, is_training = batch_data
    # reshape x and y batches so seqlen is dim 0 and batch is dim 1
    x_batch = x_batch.transpose(0,1) # switch dim 0 with dim 1
    y_batch = y_batch.transpose(0,1)
    
    hidden = repackage_hidden(hidden)    
    if is_training:
      model.zero_grad()
    output, hidden = model(x_batch, hidden)  
    output_flat = output.view(-1, ntokens)
    targets_flat = y_batch.contiguous().view(-1)  
    loss = criterion(output_flat, targets_flat)
    return loss, output_flat
    
  def on_start(state):
    state['total_train_loss'] = 0.
    state['total_test_loss'] = 0.
    state['best_val_loss'] = sys.maxsize
  
  def on_end(state):
    pass
    
  def on_sample(state):
    state['sample'].append(state['train'])
    
  def on_forward(state):
    if state['train']:
      # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
      for p in model.parameters():
        p.data.add_(-lr, p.grad.data)
      state['total_train_loss'] += state['loss'].item()
    state['total_test_loss'] += state['loss'].item()
    
  def on_start_epoch(state):
    global hidden
    state['epoch_start_time'] = time.time()
    state['total_train_loss'] = 0.
    model.train()
    hidden = model.init_hidden(args.batch_size)
    state['iterator'] = tqdm(state['iterator'], ncols=89, desc='train')
  
  def on_end_epoch(state):
    global hidden, lr
    
    model.eval()
    hidden = model.init_hidden(eval_batch_size)
    test_state = engine.test(process, tqdm(valid_loader, ncols=89, desc='test '))
    val_loss = test_state['total_test_loss'] / len(test_state['iterator'])
    train_loss = state['total_train_loss'] / len(state['iterator'])
    
    print('++ Epoch {:03d} took {:06.2f}s (lr {:5.{lrprec}f}) ++ {:s}'.format(
        state['epoch'], 
        (time.time() - state['epoch_start_time']),
        lr,
        '-'*49,
        lrprec=2 if lr >= 1 else 5))
    print('| train loss {:5.2f} | valid loss {:5.2f} | train ppl {:8.2f} | valid ppl {:8.2f}'.format( 
        train_loss, 
        val_loss,
        math.exp(train_loss),
        math.exp(val_loss)
        ))
    print('-' * 89)
    
     # Save the model if the validation loss is the best we've seen so far.
    if val_loss < state['best_val_loss']:
      state['best_val_loss'] = val_loss
      with open(args.save, 'wb') as f:
        torch.save(model, f)
    else:
      # Anneal the learning rate if no improvement has been seen in the validation dataset.
      lr = lr * args.lr_decay
  
  lr = args.lr
  engine = torchnet.engine.Engine()
  
  engine.hooks['on_start'] = on_start
  engine.hooks['on_start_epoch'] = on_start_epoch  
  engine.hooks['on_sample'] = on_sample
  engine.hooks['on_forward'] = on_forward
  engine.hooks['on_end_epoch'] = on_end_epoch
  
  dummyoptimizer = torch.optim.Adam([torch.autograd.Variable(torch.Tensor(1), requires_grad = True)])
  
  ###############################################################################
  # run training
  ###############################################################################
  
  final_state = engine.train(process, train_loader, maxepoch=args.epochs, optimizer=dummyoptimizer)
  
  # Load the best saved model.
  with open(args.save, 'rb') as f:
    model = torch.load(f)
    model.rnn.flatten_parameters()   # after load the rnn params are not a continuous chunk of memory. This makes them a continuous chunk, and will speed up forward pass
  
  # Run on test data.
  model.eval()
  hidden = model.init_hidden(eval_batch_size)
  test_state = engine.test(process, tqdm(test_loader, ncols=89, desc='test'))
  test_loss = test_state['total_test_loss'] / len(test_state['iterator'])
  print('++ End of training ++ ' + '='*67)
  print('| test loss {:5.2f} | test ppl {:8.2f} '.format(
      test_loss, 
      math.exp(test_loss)))
  print('=' * 89)

except (KeyboardInterrupt, SystemExit):
  print('Process cancelled')

