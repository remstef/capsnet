# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# coding: utf-8

import sys
if not '..' in sys.path: sys.path.append('..')

import argparse
import time
import math
import os
from tqdm import tqdm
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
import torchnet

import data
import nets.rnn
from embedding import Embedding, FastTextEmbedding, TextEmbedding, RandomEmbedding
from utils import Index, ShufflingBatchSampler, EvenlyDistributingSampler, SimpleSGD, createWrappedOptimizerClass, makeOneHot

def parseSystemArgs():
  '''
  
  '''
  parser = argparse.ArgumentParser(description='Relation Extraction')
#  parser.add_argument('--data', default='../data/semeval2010', type=str, help='location of the data corpus')
  parser.add_argument('--model', default='LSTM', type=str, help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
  parser.add_argument('--emsize', default=200, type=int, help='size of word embeddings')
  parser.add_argument('--nhid', default=200, type=int, help='number of hidden units per layer')
  parser.add_argument('--nlayers', default=2, type=int, help='number of layers')
  parser.add_argument('--lr', default=1., type=float, help='initial learning rate')
  parser.add_argument('--lr-decay', default=0.25, type=float, help='decay amount of learning learning rate if no validation improvement occurs')
  parser.add_argument('--clip', default=0.25, type=float, help='gradient clipping')
  parser.add_argument('--epochs', default=40, type=int, help='upper epoch limit')
  parser.add_argument('--batch-size', default=20, type=int, metavar='N', help='batch size') 
  parser.add_argument('--dropout', default=0.2, type=float, help='dropout applied to layers (0 = no dropout)')
  parser.add_argument('--seed', default=1111, type=int, help='random seed')
  parser.add_argument('--log-interval', default=200, type=int, metavar='N', help='report interval')
  parser.add_argument('--save', default='model.pt', type=str, help='path to save the final model')
  parser.add_argument('--init-weights', default='', type=str, help='path to initial embedding. emsize must match size of embedding')
  parser.add_argument('--chars', action='store_true', help='use character sequences instead of token sequences')
  parser.add_argument('--shuffle-batches', action='store_true', help='shuffle batches')
  parser.add_argument('--shuffle-samples', action='store_true', help='shuffle samples')
  parser.add_argument('--sequential-sampling', action='store_true', help='use samples and batches sequentially.')
  parser.add_argument('--cuda', action='store_true', help='use CUDA')
  args = parser.parse_args()
  
  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    if not args.cuda:
      print('WARNING: You have a CUDA device, so you should probably run with --cuda')

  device = torch.device('cuda' if args.cuda else 'cpu')
  setattr(args, 'device', device)

  return args


def loadData(args):
  index = Index(initwords = ['<unk>'], unkindex = 0)
  classindex = Index()
  trainset = data.SemEval2010('data/semeval2010/', subset='train.txt', index = index, classindex = classindex).to(args.device)
  index.freeze(silent = True).tofile(os.path.join('data/semeval2010/', 'vocab_chars.txt' if args.chars else 'vocab_tokens.txt'))
  classindex.freeze(silent = False).tofile(os.path.join('data/semeval2010/', 'classes.txt'))
  testset = data.SemEval2010('data/semeval2010/', subset='test.txt', index = index).to(args.device)
  
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
  
  __ItemSampler = RandomSampler if args.shuffle_samples else SequentialSampler
  __BatchSampler = BatchSampler if args.sequential_sampling else EvenlyDistributingSampler  
  train_loader = torch.utils.data.DataLoader(trainset, batch_sampler = ShufflingBatchSampler(__BatchSampler(__ItemSampler(trainset), batch_size=args.batch_size, drop_last = True), shuffle = args.shuffle_batches, seed = args.seed), num_workers = 0)
  test_loader = torch.utils.data.DataLoader(testset, batch_sampler = __BatchSampler(__ItemSampler(testset), batch_size=args.batch_size, drop_last = True), num_workers = 0)

  print(__ItemSampler.__name__)
  print(__BatchSampler.__name__)
  print('Shuffle training batches: ', args.shuffle_batches)

  setattr(args, 'maxseqlen', trainset.maxseqlen)
  setattr(args, 'index', index)
  setattr(args, 'index', classindex)
  setattr(args, 'ntoken', len(index))
  setattr(args, 'nclasses', len(classindex))
  setattr(args, 'trainloader', train_loader)
  setattr(args, 'testloader', test_loader)
  setattr(args, 'preembweights', preemb_weights)

  return args

def buildModel(args):

  model = nets.rnn.RNN_CLASSIFY_simple(
      ntoken = args.ntoken,
      nhid = args.nhid, 
      nclasses = args.nclasses
      ).to(args.device)
  criterion = torch.nn.NLLLoss() # CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr =args.lr)

  print(model)
  print(criterion)
  print(optimizer)
  
  setattr(args, 'model', model)
  setattr(args, 'criterion', criterion)
  setattr(args, 'optimizer', optimizer)
  
  return args

###############################################################################
# Process one batch
###############################################################################
def getprocessfun(args):
  model = args.model
  
  def process(batch_data):
    x_batch, y_batch, seqlengths, hidden_before, is_training = batch_data
    x_batch_one_hot = makeOneHot(x_batch, args.ntoken)
    x_batch_one_hot = x_batch_one_hot.transpose(0,1).to(args.device) # switch dim 0 with dim 1 => x_batch_one_hot = seqlen x batch x ntoken

    hidden = model.init_hidden(x_batch_one_hot.size(1))

    if is_training:
      model.zero_grad()
    for i in range(x_batch_one_hot.size(0)):
      output, hidden = model(x_batch_one_hot[i], hidden)

    loss = args.criterion(output, y_batch)    
    
    return loss, (output, hidden)
  
  return process


if __name__ == '__main__':
 
  try:
    
    args = parseSystemArgs()
    args = loadData(args)
    args = buildModel(args)  
    process = getprocessfun(args)
    model = args.model

    ###############################################################################
    # Set up Engine
    ###############################################################################
          
    def on_start(state):
      state['train_loss'] = 0.
      state['test_loss'] = 0.
      state['train_loss_per_interval'] = 0.
    
    def on_end(state):
      pass
      
    def on_sample(state):
      state['sample'].append(state['hidden'])
      state['sample'].append(state['train'])
      state['batch_start_time'] = time.time()
      state['hidden'] = model.init_hidden(args.batch_size)
      
    def on_forward(state):
      # TODO: track meters
      outputs, hidden = state['output']
      state['hidden'] = hidden
      loss_val = state['loss'].item()
      state['train_loss'] += loss_val
      state['test_loss'] += loss_val
      state['train_loss_per_interval'] += loss_val
      if state['train']:        
        t = state['t']
        t_epoch = t % len(state['iterator'])
        if t_epoch % args.log_interval == 0 and t_epoch > 0:
          epoch = state['epoch']
          maxepoch = state['maxepoch']
          t_epoch = t % len(state['iterator'])
          cur_loss = state['train_loss_per_interval'] / args.log_interval
          elapsed = time.time() - state['batch_start_time']
          tqdm.write('| epoch {:3d} / {:3d} | batch {:5d} / {:5d} | ms/batch {:5.2f} | loss {:5.2f}'.format(
              epoch+1,
              maxepoch,
              t_epoch,
              len(state['iterator']),
              (elapsed * 1000) / args.log_interval, 
              cur_loss
              ))
          state['train_loss_per_interval'] = 0.
      
    def on_start_epoch(state):
      state['epoch_start_time'] = time.time()
      state['train_loss'] = 0.
      state['train_loss_per_interval'] = 0.      
      state['iterator'] = tqdm(state['iterator'], ncols=89, desc='train')
      model.train()
      state['hidden'] = model.init_hidden(args.batch_size)
    
    def on_end_epoch(state):
      model.eval()
      test_state = engine.test(process, tqdm(args.validloader, ncols=89, desc='test '))
      val_loss = test_state['test_loss'] / len(test_state['iterator'])
      train_loss = state['train_loss'] / len(state['iterator'])
      
      print(
          '''
          ++ Epoch {:03d} took {:06.2f}s (lr {:5.{lrprec}f}) ++ {:s}
          | train loss {:5.2f} | valid loss {:5.2f} | train ppl {:8.2f} | valid ppl {:8.2f}
          {:s}
          '''.format(
          state['epoch'], 
          (time.time() - state['epoch_start_time']),
          args.optimizer.getLearningRate(),
          '-'*(49 if args.optimizer.getLearningRate() >= 1 else 47),
          train_loss, 
          val_loss,
          math.exp(train_loss),
          math.exp(val_loss),
          '-' * 89,
          lrprec=2 if args.optimizer.getLearningRate() >= 1 else 5))
  
    # define engine 
    engine = torchnet.engine.Engine()  
    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch  
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end_epoch'] = on_end_epoch
      
    ###############################################################################
    # run training
    ###############################################################################
    
    final_state = engine.train(process, args.trainloader, maxepoch=args.epochs, optimizer=args.optimizer)
      
  except (KeyboardInterrupt, SystemExit):
    print('Process cancelled')
 
#import string   
#all_letters = string.ascii_letters + " .,;'"
#n_letters = len(all_letters)
#    
#def letterToIndex(letter):
#    return all_letters.find(letter)
#
## Just for demonstration, turn a letter into a <1 x n_letters> Tensor
#def letterToTensor(letter):
#    tensor = torch.zeros(1, n_letters)
#    tensor[0][letterToIndex(letter)] = 1
#    return tensor
#
## Turn a line into a <line_length x 1 x n_letters>,
## or an array of one-hot letter vectors
#def lineToTensor(line):
#    tensor = torch.zeros(len(line), 1, n_letters)
#    for li, letter in enumerate(line):
#        tensor[li][0][letterToIndex(letter)] = 1
#    return tensor
#
#print(letterToTensor('J'))
#
#print(lineToTensor('Jones').size())
  