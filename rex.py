# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# coding: utf-8

import sys
if not '..' in sys.path: sys.path.append('..')

import argparse
import time
import os
from tqdm import tqdm
import sklearn.metrics
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
import torchnet


import data
import utils
import nets.rnn
import embedding


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
  parser.add_argument('--engine', action='store_true', help='use torchnet engine for traininge and testing.')
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
  index = utils.Index(initwords = ['<unk>'], unkindex = 0)
  classindex =utils.Index()
  trainset = data.SemEval2010('data/semeval2010/', subset='train.txt', index = index, classindex = classindex).to(args.device)
  index.freeze(silent = True).tofile(os.path.join('data/semeval2010/', 'vocab_chars.txt' if args.chars else 'vocab_tokens.txt'))
  classindex.freeze(silent = False).tofile(os.path.join('data/semeval2010/', 'classes.txt'))
  testset = data.SemEval2010('data/semeval2010/', subset='test.txt', index = index).to(args.device)
  
  # load pre embedding
  if args.init_weights:
    # determine type of embedding by checking it's suffix
    if args.init_weights.endswith('bin'):
      preemb = embedding.FastTextEmbedding(args.init_weights, normalize = True).load()
      if args.emsize != preemb.dim():
        raise ValueError('emsize must match embedding size. Expected %d but got %d)' % (args.emsize, preemb.dim()))
    elif args.init_weights.endswith('txt'):
      preemb = embedding.TextEmbedding(args.init_weights, vectordim = args.emsize).load(normalize = True)
    elif args.init_weights.endswith('rand'):
      preemb = embedding.RandomEmbedding(vectordim = args.emsize)
    else:
      raise ValueError('Type of embedding cannot be inferred.')
    preemb = embedding.Embedding.filteredEmbedding(index.vocabulary(), preemb, fillmissing = True)
    preemb_weights = torch.Tensor(preemb.weights)
  else:
    preemb_weights = None
  
  __ItemSampler = RandomSampler if args.shuffle_samples else SequentialSampler
  __BatchSampler = BatchSampler if args.sequential_sampling else utils.EvenlyDistributingSampler  
  train_loader = torch.utils.data.DataLoader(trainset, batch_sampler = utils.ShufflingBatchSampler(__BatchSampler(__ItemSampler(trainset), batch_size=args.batch_size, drop_last = True), shuffle = args.shuffle_batches, seed = args.seed), num_workers = 0)
  test_loader = torch.utils.data.DataLoader(testset, batch_sampler = __BatchSampler(__ItemSampler(testset), batch_size=args.batch_size, drop_last = True), num_workers = 0)

  print(__ItemSampler.__name__)
  print(__BatchSampler.__name__)
  print('Shuffle training batches: ', args.shuffle_batches)

  setattr(args, 'maxseqlen', trainset.maxseqlen)
  setattr(args, 'index', index)
  setattr(args, 'classindex', classindex)
  setattr(args, 'ntoken', len(index))
  setattr(args, 'nclasses', len(classindex))
  setattr(args, 'trainloader', train_loader)
  setattr(args, 'testloader', test_loader)
  setattr(args, 'preembweights', preemb_weights)
  setattr(args, 'confusion_meter', torchnet.meter.ConfusionMeter(len(classindex), normalized=True))

  return args

def buildModel(args):

  model = nets.rnn.RNN_CLASSIFY_linear(
      ntoken = args.ntoken,
      nhid = args.nhid, 
      nclasses = args.nclasses
      ).to(args.device)
  criterion = torch.nn.NLLLoss() # CrossEntropyLoss()
  optimizer = utils.createWrappedOptimizerClass(torch.optim.SGD)(model.parameters(), lr =args.lr, clip=None)

  print(model)
  print(criterion)
  print(optimizer)
  
  setattr(args, 'model', model)
  setattr(args, 'criterion', criterion)
  setattr(args, 'optimizer', optimizer)
  
  return args

def message_status_interval(message, epoch, max_epoch, batch_i, nbatches, batch_start_time, log_interval, train_loss_interval, predictions, targets):
  return '| epoch {:3d} / {:3d} | batch {:5d} / {:5d} | ms/batch {:5.2f} | loss {:5.2f}'.format(
      epoch,
      max_epoch,
      batch_i,
      nbatches,
      ((time.time() - batch_start_time) * 1000) / log_interval, 
      train_loss_interval)

def message_status_endepoch(message, epoch, epoch_start_time, learning_rate, train_loss, test_loss, predictions, targets):
  p = sklearn.metrics.precision_score(targets, predictions, average='micro')
  r = sklearn.metrics.recall_score(targets, predictions, average='micro')
  f = sklearn.metrics.f1_score(targets, predictions, average='micro')
  return '''\
++ Epoch {:03d} took {:06.2f}s (lr {:5.{lrprec}f}) ++ {:s}
| train loss {:5.2f} | test loss {:5.2f} | p {:4.2f} | r {:4.2f} | f1 {:4.2f}
{:s}\
'''.format(
      epoch, 
      (time.time() - epoch_start_time),
      learning_rate,
      '-'*(49 if learning_rate >= 1 else 47),
      train_loss, 
      test_loss,
      p, r, f,
      '-' * 89,
      lrprec=2 if learning_rate >= 1 else 5)

def getpredictions(batch_logprobs):
  return batch_logprobs.max(dim=1)[1]  

###############################################################################
# Functions which process one batch
###############################################################################
def getprocessfun(args):
  model = args.model
  
  def process(batch_data):
    x_batch, y_batch, seqlengths, hidden_before, is_training = batch_data
    x_batch_one_hot = utils.makeOneHot(x_batch, args.ntoken)
    x_batch_one_hot = x_batch_one_hot.transpose(0,1) # switch dim 0 with dim 1 => x_batch_one_hot = seqlen x batch x ntoken

    hidden = model.init_hidden(x_batch_one_hot.size(1))

    for i in range(x_batch_one_hot.size(0)):
      outputs, hidden = model(x_batch_one_hot[i], hidden)

    loss = args.criterion(outputs, y_batch)    
    
    return loss, (outputs, hidden)
  
  return process

###############################################################################
# Run in Pipeline mode
###############################################################################
def pipeline(args):
    
  def evaluate(args, dloader):
    model = args.model
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    predictions = []
    targets = []
    
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
      for batch, batch_data in enumerate(tqdm(dloader, ncols=89, desc = 'Test ')):   
        batch_data.append(hidden)
        batch_data.append(False)
        loss, (outputs, hidden) = process(batch_data)
        # keep track of some scores
        total_loss += args.batch_size * loss.item()
        args.confusion_meter.add(outputs.data, batch_data[1])
        predictions.extend(getpredictions(outputs.data).tolist())
        targets.extend(batch_data[1].tolist())
        
    test_loss = total_loss / (len(dloader) * args.batch_size )
    return test_loss, predictions, targets
  
  
  def train(args):
    model = args.model
    # Turn on training mode which enables dropout.
    model.train()
    train_loss = 0.
    interval_loss = 0.
    predictions = []
    targets = []
    
    hidden = model.init_hidden(args.batch_size)    
    for batch, batch_data in enumerate(tqdm(args.trainloader, ncols=89, desc='Train')):
      batch_start_time = time.time()
      model.zero_grad()
      loss, (outputs, hidden) = process(batch_data + [hidden, True])
      loss.backward()
      args.optimizer.step()
      # track some scores
      train_loss += loss.item()
      interval_loss += loss.item()
      args.confusion_meter.add(outputs.data, batch_data[1])
      predictions.extend(getpredictions(outputs.data).tolist())
      targets.extend(batch_data[1].tolist())
  
      if batch % args.log_interval == 0 and batch > 0:
        cur_loss = train_loss / args.log_interval
        tqdm.write(message_status_interval('Current Status:', epoch+1, args.epochs, batch, len(args.trainloader), batch_start_time, args.log_interval, cur_loss, predictions, targets))
        interval_loss = 0.
      train_loss = train_loss / (len(args.trainloader) * args.batch_size)
    return train_loss, predictions, targets

  ###
  # Run pipeline
  ###
  process = getprocessfun(args)
  
  for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train_loss, _, _ = train(args)
    test_loss, predictions, targets = evaluate(args, args.testloader)
    print(message_status_endepoch('', epoch+1, epoch_start_time, args.optimizer.getLearningRate(), train_loss, test_loss, predictions, targets))

###############################################################################
# Run in Engine mode
###############################################################################
def engine(args):

    ###########################################################################
    # Set up Engine
    ###########################################################################
    def on_start(state):
      state['train_loss'] = 0.
      state['test_loss'] = 0.
      state['train_loss_per_interval'] = 0.
      state['hidden'] = model.init_hidden(args.batch_size)
    
    def on_end(state):
      pass
      
    def on_sample(state):
      state['sample'].append(state['hidden'])
      state['sample'].append(state['train'])
      state['batch_start_time'] = time.time()
      
    def on_forward(state):
      # TODO: meters for tracking performance
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
          state['train_loss_per_interval'] = 0.
          tqdm.write(message_status_interval('Current status', epoch+1, maxepoch, t_epoch, len(state['iterator']), state['batch_start_time'], args.log_interval, cur_loss))
      
    def on_start_epoch(state):
      state['epoch_start_time'] = time.time()
      state['train_loss'] = 0.
      state['train_loss_per_interval'] = 0.      
      state['iterator'] = tqdm(state['iterator'], ncols=89, desc='train')
      model.train()
      state['hidden'] = model.init_hidden(args.batch_size)
    
    def on_end_epoch(state):
      model.eval()
      test_state = engine.test(process, tqdm(args.testloader, ncols=89, desc='test '))
      test_loss = test_state['test_loss'] / len(test_state['iterator'])
      train_loss = state['train_loss'] / len(state['iterator'])
      print(message_status_endepoch('End of epoch', state['epoch'], state['epoch_start_time'], args.optimizer.getLearningRate(), train_loss, test_loss))
  
    # define engine 
    engine = torchnet.engine.Engine()  
    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch  
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end_epoch'] = on_end_epoch
      
    ###########################################################################
    # run training
    ###########################################################################

    process = getprocessfun(args)
    model = args.model
    
    engine.train(process, args.trainloader, maxepoch=args.epochs, optimizer=args.optimizer)


if __name__ == '__main__':
 
  try:
    
    args = parseSystemArgs()
    args = loadData(args)
    args = buildModel(args)  
    
    if args.engine:
      print('Running in torchnet engine.')
      engine(args)
    else:
      pipeline(args)
    
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
  