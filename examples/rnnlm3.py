# -*- coding: utf-8 -*-

# coding: utf-8

import sys
if not '..' in sys.path: sys.path.append('..')

import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

from data import WikiSentences
from utils import Index, RandomBatchSampler
from embedding import Embedding, FastTextEmbedding, TextEmbedding, RandomEmbedding

import rnnlm_net as model

    
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
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
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

train_loader = torch.utils.data.DataLoader(train_, batch_sampler = RandomBatchSampler(torch.utils.data.sampler.SequentialSampler(train_), batch_size=args.batch_size, drop_last = True), num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_, batch_sampler = RandomBatchSampler(torch.utils.data.sampler.SequentialSampler(test_), batch_size=eval_batch_size, drop_last = True), num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid_, batch_sampler = RandomBatchSampler(torch.utils.data.sampler.SequentialSampler(valid_), batch_size=eval_batch_size, drop_last = True), num_workers = 0)

###############################################################################
# Build the model
###############################################################################
ntokens = len(index)
model = model.RNNModel(
    rnn_type = args.model, 
    ntoken = ntokens, 
    ninp = args.emsize, 
    nhid = args.nhid, 
    nlayers = args.nlayers, 
    dropout = args.dropout, 
    tie_weights = args.tied, 
    init_em_weights = preemb_weights, 
    train_em_weights = True).to(device)

criterion = nn.CrossEntropyLoss()
hidden = model.init_hidden(args.batch_size)
lr = args.lr

###############################################################################
# Set up Engine
###############################################################################
 
import torchnet
from tqdm import tqdm
 
engine = torchnet.engine.Engine()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def process(batch_data):
  global hidden, lr
  
  x_batch, y_batch, is_training = batch_data
  # reshape x_batch so seqlen is dim 0 and batch is dim 1
  x_batch = x_batch.transpose(0,1) # switch dim 0 with dim 1
  # reshape y_batch so we get a 1d tensor of length seqlen x batch that matches with x_batch
  y_batch = y_batch.transpose(0,1)

  data, targets = x_batch, y_batch

  hidden = repackage_hidden(hidden)    
  if is_training:
    model.zero_grad()
  output, hidden = model(data, hidden)  
  output_flat = output.view(-1, ntokens)
  targets_flat = targets.contiguous().view(-1)
  
  loss = criterion(output_flat, targets_flat)

  return loss, output_flat

  
def on_start(state):
  pass
  
def on_sample(state):
  state['sample'].append(state['train'])
  
def on_forward(state):
  # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
  torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
  for p in model.parameters():
      p.data.add_(-lr, p.grad.data)
  state['total_loss'] = state['loss'].item()
  
def on_start_epoch(state):
  global hidden
  state['epoch_start_time'] = time.time()
  model.train()
  hidden = model.init_hidden(args.batch_size)
  state['iterator'] = tqdm(state['iterator'])

def on_end_epoch(state):
  global hidden
  model.eval()
  hidden = model.init_hidden(eval_batch_size)
  engine.test(process, valid_loader)
  
  total_loss = state['total_loss']
  print('-' * 89)
  print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
      state['epoch'], 
      (time.time() - state['epoch_start_time']), 
      total_loss, 
      math.exp(total_loss)
      ))
  print('-' * 89)

engine.hooks['on_start'] = on_start
engine.hooks['on_start_epoch'] = on_start_epoch  
engine.hooks['on_sample'] = on_sample
engine.hooks['on_forward'] = on_forward
engine.hooks['on_end_epoch'] = on_end_epoch

dummyoptimizer = torch.optim.Adam([torch.autograd.Variable(torch.Tensor(1), requires_grad = True)])

engine.train(process, train_loader, maxepoch=args.epochs, optimizer=dummyoptimizer)

model.eval()
hidden = model.init_hidden(eval_batch_size)
engine.test(process, test_loader)

