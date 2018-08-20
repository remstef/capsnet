# -*- coding: utf-8 -*-

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

'''

'''
def parseSystemArgs():
  '''
  
  '''
  parser = argparse.ArgumentParser(description='Relation Extraction')
#  parser.add_argument('--data', default='../data/semeval2010', type=str, help='location of the data corpus')
#  parser.add_argument('--rnntype', default='LSTM', type=str, help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
  parser.add_argument('--optim', default='SGD', type=str, help='type of optimizer (SGD, Adam, Adagrad, ASGD, SimpleSGD)')
  parser.add_argument('--loss-criterion', default='NLLLoss', type=str, help='type of loss function to use (NLLLoss, CrossEntropyLoss)')
  parser.add_argument('--emsize', default=300, type=int, help='size of word embeddings')
  parser.add_argument('--posiemsize', default=5, type=int, help='size of the position embeddings')
  parser.add_argument('--classemsize', default=4, type=int, help='size of label embeddings')
  parser.add_argument('--maxdist', default=60, type=int, help='maximum distance for position embeddings')
  parser.add_argument('--windowsize', default=3, type=int, help='number of convolution filters to apply')
  parser.add_argument('--numconvfilters', default=200, type=int, help='size of the moving convolutional window')
  parser.add_argument('--conv-windowsize', default=1, type=int, help='size of the moving convolutional window')
  parser.add_argument('--conv-activation', default='ReLU', type=str, help='activation function to use after convolutinal layer (ReLU, Tanh)')
  parser.add_argument('--nhid', default=200, type=int, help='size of hidden layer')
  parser.add_argument('--lr', default=1., type=float, help='initial learning rate')
#  parser.add_argument('--lr-decay', default=0.25, type=float, help='decay amount of learning learning rate if no validation improvement occurs')
  parser.add_argument('--wdecay', default=1.2e-6, type=float, help='weight decay applied to all weights')
  parser.add_argument('--clip', default=0.25, type=float, help='gradient clipping')
  parser.add_argument('--epochs', default=40, type=int, help='upper epoch limit')
  parser.add_argument('--batch-size', default=50, type=int, metavar='N', help='batch size') 
  parser.add_argument('--dropout', default=0.2, type=float, help='dropout applied to layers (0 = no dropout)')
  parser.add_argument('--seed', default=1111, type=int, help='random seed')
  parser.add_argument('--log-interval', default=60, type=int, metavar='N', help='report interval')
  parser.add_argument('--save', default='model.pt', type=str, help='path to save the final model')
  parser.add_argument('--init-word-weights', default='', type=str, help='path to initial word embedding; emsize must match size of embedding')
  parser.add_argument('--fix-word-weights', action='store_true', help='Specify if the word embedding should be excluded from further training')
  parser.add_argument('--shuffle-batches', action='store_true', help='shuffle batches')
  parser.add_argument('--shuffle-samples', action='store_true', help='shuffle samples')
  parser.add_argument('--cuda', action='store_true', help='use CUDA')
  parser.add_argument('--engine', action='store_true', help='use torchnet engine for training and testing.')
  args = parser.parse_args()
    
  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    if not args.cuda:
      print('WARNING: You have a CUDA device, so you should probably run with --cuda')

  args.device = torch.device('cuda' if args.cuda else 'cpu')
  return args
  
def loadData(args):
  index = utils.Index(initwords = ['<unk>'], unkindex = 0)
  
  trainset = data.SemEval2010('data/semeval2010/', subset='train.txt', nlines = None, index = index, maxdist = args.maxdist, nbos = args.windowsize // 2, neos = args.windowsize // 2).to(args.device)
  
  trainset.index.freeze(silent = True).tofile(os.path.join('data/semeval2010/', 'vocab.txt'))
  trainset.posiindex.freeze(silent = True).tofile(os.path.join('data/semeval2010/', 'position-index.txt'))
  trainset.classindex.freeze(silent = False).tofile(os.path.join('data/semeval2010/', 'classes.txt'))
  trainset.rclassindex.freeze(silent = False).tofile(os.path.join('data/semeval2010/', 'classes-rel.txt'))
  trainset.dclassindex.freeze(silent = False).tofile(os.path.join('data/semeval2010/', 'classes-direction.txt'))
  trainset.eclassindex.freeze(silent = False).tofile(os.path.join('data/semeval2010/', 'classes-entity.txt'))
  
  testset = data.SemEval2010('data/semeval2010/', subset='test.txt', nlines = None, maxseqlen = trainset.maxseqlen, maxentlen = trainset.maxentlen, index = index, nbos = args.windowsize // 2, neos = args.windowsize // 2, maxdist=args.maxdist, posiindex = trainset.posiindex, classindex = trainset.classindex, rclassindex = trainset.rclassindex, dclassindex = trainset.dclassindex, eclassindex = trainset.eclassindex).to(args.device)
  
  print('train: ' + str(trainset))
  print('test: ' + str(testset))
  
  # load pre embedding
  if args.init_word_weights:
    # determine type of embedding by checking it's suffix
    if args.init_word_weights.endswith('bin'):
      preemb = embedding.FastTextEmbedding(args.init_word_weights, normalize = True).load()
      if args.emsize != preemb.dim():
        raise ValueError(f'emsize must match embedding size. Expected {args.emsize:d} but got {preemb.dim():d}')
    elif args.init_word_weights.endswith('txt'):
      preemb = embedding.TextEmbedding(args.init_word_weights, vectordim = args.emsize).load(normalize = True)
    elif args.init_word_weights.endswith('rand'):
      preemb = embedding.RandomEmbedding(vectordim = args.emsize)
    else:
      raise ValueError('Type of embedding cannot be inferred.')
    preemb = embedding.Embedding.filteredEmbedding(index.vocabulary(), preemb, fillmissing = True)
    preemb_weights = torch.Tensor(preemb.weights)
  else:
    preemb_weights = None
  
  __ItemSampler = RandomSampler if args.shuffle_samples else SequentialSampler
  train_loader = torch.utils.data.DataLoader(trainset, batch_sampler = utils.ShufflingBatchSampler(BatchSampler(__ItemSampler(trainset), batch_size=args.batch_size, drop_last = False), shuffle = args.shuffle_batches, seed = args.seed), num_workers = 0)
  test_loader = torch.utils.data.DataLoader(testset, batch_sampler = BatchSampler(__ItemSampler(testset), batch_size=args.batch_size, drop_last = False), num_workers = 0)

  print(__ItemSampler.__name__)
  print('Shuffle training batches: ', args.shuffle_batches)

  args.maxseqlen = trainset.maxseqlen
  args.maxentlen = trainset.maxentlen
  args.index = index
  args.posiindex = trainset.posiindex
  args.classindex = trainset.classindex
  args.rclassindex = trainset.rclassindex
  args.dclassindex = trainset.dclassindex
  args.eclassindex = trainset.eclassindex
  args.ntoken = len(index)
  args.nclasses = len(trainset.classindex)
  args.trainloader = train_loader
  args.testloader = test_loader
  args.preembweights = preemb_weights
  args.confusion_meter = torchnet.meter.ConfusionMeter(len(trainset.classindex), normalized=True)

  return args

def buildModel(args):
  '''
  Build the model, processing function for one batch and loss criterion
  '''

#  model = nets.rnn.RNN_CLASSIFY_linear(
#      ntoken = args.ntoken,
#      nhid = args.nhid, 
#      nclasses = args.nclasses
#      ).to(args.device)
  
#  def process_linear_rnn(batch_data):
#
#    # unpack data that was already prepared (batch_first_dim)
#    ith_sample, seq, seqlen, relposi_vec_e1, relposi_vec_e2, offs_left, offs_e1, offs_mid, offs_e2, offs_right, seq_e1, seqlen_e1, seq_e2, seqlen_e2, label, e1label, e2label, rlabel, dlabel, h, train = batch_data
#    assert ith_sample.size(0) == args.batch_size, f"That's odd, batch dimension shoudl be {args.batch_size:d} but is {ith_sample.size(0)}."
#    
#    x_batch_one_hot = utils.makeOneHot(seq, args.ntoken)
#    x_batch_one_hot = x_batch_one_hot.transpose(0,1) # switch dim 0 with dim 1 => x_batch_one_hot = seqlen x batch x ntoken
#    targets = label
#
#    hidden = model.init_hidden(x_batch_one_hot.size(1))
#
#    for i in range(x_batch_one_hot.size(0)):
#      outputs, hidden = model(x_batch_one_hot[i], hidden)
#      
#    loss = criterion(outputs, targets)
#    
#    predictions = getpredictions(outputs.data)
#    return loss, (outputs, predictions, targets, hidden)
  
  model = nets.rnn.ReClass(
      ntoken              = args.ntoken,
      nclasses            = args.nclasses,
      maxdist             = args.maxdist,
      maxseqlength        = args.maxseqlen,
      maxentlength        = args.maxentlen,
      window_size         = args.windowsize,
      emsizeword          = args.emsize,
      emsizeposi          = args.posiemsize,
      emsizeclass         = args.classemsize,
      nhid                = args.nhid,
      numconvfilters      = args.numconvfilters,
      convwindow          = args.conv_windowsize,
      dropout             = args.dropout,
      conv_activation     = args.conv_activation,
      weightsword         = args.preembweights,
      fix_emword          = args.fix_word_weights
      ).to(args.device)
  
  if not args.loss_criterion in ['NLLLoss', 'CrossEntropyLoss']:
    raise ValueError( '''Invalid option `%s` for 'loss-criterion' was supplied.''' % args.loss_criterion)
  criterion = getattr(torch.nn, args.loss_criterion)()
  
  def process(batch_data):
    # unpack data that was already prepared (batch_first_dim)
    ith_sample, sample_id, seq, seqlen, relposi_vec_e1, relposi_vec_e2, offs_left, offs_e1, offs_mid, offs_e2, offs_right, seq_e1, seqlen_e1, seq_e2, seqlen_e2, label, e1label, e2label, rlabel, dlabel, train = batch_data
    # assert ith_sample.size(0) == args.batch_size, f"That's odd, batch dimension should be {args.batch_size:d} but is {ith_sample.size(0)}."
    targets = label
    
    outputs, labelweights = model(seq, seqlen, seq_e1, seqlen_e1, seq_e2, seqlen_e2, offs_e1, offs_e2, relposi_vec_e1, relposi_vec_e2)
      
    loss = criterion(outputs, targets)
    
    predictions = getpredictions(outputs.data)
    return loss, (sample_id, outputs, predictions, targets)
    
  print(model)
  print(criterion)
  
  args.model = model
  args.modelprocessfun = process
  args.criterion = criterion
  
  return args

def getOptimizer(args):
  if args.optim == 'SimpleSGD':
    Optimizer__ = utils.SimpleSGD
  elif not args.optim in [ 'SGD', 'Adam', 'ASGD', 'Adagrad' ]:
    raise ValueError( '''Invalid option `%s` for 'optimizer' was supplied.''' % args.optim)
  else:
    Optimizer__ = getattr(torch.optim, args.optim)
  optimizer = utils.createWrappedOptimizerClass(Optimizer__)(args.model.parameters(), lr =args.lr, clip=None, weight_decay=args.wdecay)  
  args.optimizer = optimizer
  return args

def getpredictions(batch_logprobs):
  return batch_logprobs.max(dim=1)[1]

def getscores(targets, predictions):
  ''' official scoring (label without direction) + direction_errors as single error class
  my $P = $$confMatrix{$labelAnswer}{$labelAnswer} / ($$allLabelsProposed{$labelAnswer} + $wrongDirectionCnt);
  my $R = $$confMatrix{$labelAnswer}{$labelAnswer} / $$allLabelsAnswer{$labelAnswer};
  my $F1 = 2 * $P * $R / ($P + $R);
  '''
  vals = {
      'A': sklearn.metrics.accuracy_score(targets, predictions),
      'P': sklearn.metrics.precision_score(targets, predictions, average='macro'),
      'R': sklearn.metrics.recall_score(targets, predictions, average='macro'),
      'F': sklearn.metrics.f1_score(targets, predictions, average='macro')
      }
  return vals

def message_status_interval(message, epoch, max_epoch, batch_i, nbatches, batch_start_time, log_interval, train_loss_interval, scores):
  scoreline = ' | '.join(['{:s} {:6.4f}'.format(k, v) for k, v in scores.items()])
  return '''\
| Status: Batch {:d} - {:d} / {:d} | Epoch {:d} / {:d} | ms/batch {:5.2f} 
|   +-- Training loss {:.10f}
|   +-- {:s}
|\
'''.format(
      batch_i - args.log_interval,
      batch_i,
      nbatches,
      epoch,
      max_epoch,
      ((time.time() - batch_start_time) * 1000) / log_interval, 
      train_loss_interval,
      scoreline)

def message_status_endepoch(message, epoch, epoch_start_time, learning_rate, train_loss, test_loss, scores):
  scoreline = ' | '.join(['{:s} {:6.4f}'.format(k, v) for k, v in scores.items()])
  return '''\
|
|{:s}
| Epoch {:03d} took {:06.2f}s
|   +-- Learing rate {:10.6f}
|   +-- Loss (train) {:.10f}
|   +-- Loss (test)  {:.10f}
|   +-- {:s}
|{:s}
|
|\
'''.format(
      '=' * 88,
      epoch, 
      (time.time() - epoch_start_time),
      learning_rate,
      train_loss, 
      test_loss,
      scoreline,
      '=' * 88)

def loadmodel(args):
  with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
  return model

def savemodel(args):
  with open(args.save, 'wb') as f:
    torch.save(args.model, f)
    
def savepredictions(args, ids, logprobs, predictions, targets, scores):
  outfile = f'{args.save:s}.predictions.tsv'
  assert len(ids) == len(logprobs) == len(predictions) == len(targets), f'Something is wrong, number of samples and number of predicions are different: {len(ids):s} {len(logprobs):s} {len(predictions):s} {len(targets):s}'
  with open(outfile, 'w') as f:
    print('# ' + ' | '.join(['{:s} {:6.4f}'.format(k, v) for k, v in scores.items()]), file=f)
    for i in range(len(ids)):
      pred_classlabel = args.classindex[predictions[i]]
      true_classlabel = args.classindex[targets[i]]
      correct = int(predictions[i] == targets[i])
      print(f'{ids[i]:d}\t{pred_classlabel:s}\t{true_classlabel:s}\t{correct:d}\t{predictions[i]:d}\t{targets[i]:d}\t{logprobs[i]:}', file=f)

###############################################################################
# Run in Pipeline mode
###############################################################################
def pipeline(args):
    
  def evaluate(args, dloader):
    model = args.model
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ids = []
    predictions = []
    logprobs = []
    targets = []
    
    with torch.no_grad():
      for batch, batch_data in enumerate(tqdm(dloader, ncols=89, desc = 'Test ')):
        batch_data.append(False)
        loss, (sampleids, outputs, predictions_, targets_) = process(batch_data)
        # keep track of some scores
        total_loss += args.batch_size * loss.item()
        args.confusion_meter.add(outputs.data, targets_)
        ids.extend(sampleids.tolist())
        logprobs.extend(outputs.data.tolist())
        predictions.extend(predictions_.tolist())
        targets.extend(targets_.tolist())
        
    test_loss = total_loss / (len(dloader) * args.batch_size )
    return test_loss, ids, logprobs, predictions, targets
  
  
  def train(args):
    model = args.model
    # Turn on training mode which enables dropout.
    model.train()
    train_loss = 0.
    interval_loss = 0.
    predictions = []
    targets = []
    
    for batch, batch_data in enumerate(tqdm(args.trainloader, ncols=89, desc='Train')):
      batch_start_time = time.time()
      model.zero_grad()
      loss, (_, outputs, predictions_, targets_) = process(batch_data + [ True ])
      loss.backward()
      args.optimizer.step()
      # track some scores
      train_loss += loss.item()
      interval_loss += loss.item()
      
      args.confusion_meter.add(outputs.data, targets_)
      predictions.extend(predictions_.tolist())
      targets.extend(targets_.tolist())
  
      if batch % args.log_interval == 0 and batch > 0:
        cur_loss = train_loss / args.log_interval
        scores = getscores(targets, predictions)
        tqdm.write(message_status_interval('Current Status:', epoch+1, args.epochs, batch, len(args.trainloader), batch_start_time, args.log_interval, cur_loss, scores))
        interval_loss = 0.
      train_loss = train_loss / (len(args.trainloader) * args.batch_size)
    return train_loss, predictions, targets

  ###
  # Run pipeline
  ###
  best_test_val = 0
  process = args.modelprocessfun
  for epoch in tqdm(range(args.epochs), ncols=89, desc = 'Epochs'):
    epoch_start_time = time.time()
    train_loss, _, _ = train(args)
    test_loss, sampleids, logprobs, predictions, targets = evaluate(args, args.testloader)
    scores = getscores(targets, predictions)
    tqdm.write(message_status_endepoch('', epoch+1, epoch_start_time, args.optimizer.getLearningRate(), train_loss, test_loss, scores))
    if best_test_val < scores['F']:
      tqdm.write('> Saving model and prediction results...')
      savemodel(args)
      savepredictions(args, sampleids, logprobs, predictions, targets, scores)
      best_test_val = scores['F']
      tqdm.write('> ... Finished saving')

###############################################################################
# Run in Engine mode
###############################################################################
def engine(args):
  
    raise NotImplementedError('This method is not in sync with the pipeline method and throws errors.')

    ###########################################################################
    # Set up Engine
    ###########################################################################
    def on_start(state):
      state['train_loss'] = 0.
      state['test_loss'] = 0.
      state['train_loss_per_interval'] = 0.
    
    def on_end(state):
      pass
      
    def on_sample(state):
      state['sample'].append(state['train'])
      state['batch_start_time'] = time.time()
      
    def on_forward(state):
      outputs, predictions, targets  = state['output']
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

    process = args.modelprocessfun
    model = args.model
    engine.train(process, args.trainloader, maxepoch=args.epochs, optimizer=args.optimizer)

if __name__ == '__main__':
 
  try:
    args = parseSystemArgs()
    args = loadData(args)
    args = buildModel(args)  
    args = getOptimizer(args)    
    if args.engine:
      print('Running in torchnet engine.')
      engine(args)
    else:
      pipeline(args)    
  except (KeyboardInterrupt, SystemExit):
    print('Process cancelled')
   