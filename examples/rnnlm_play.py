# -*- coding: utf-8 -*-

import sys
import os
if not '..' in sys.path: sys.path.append('..')

###############################################################################
# Language Modeling on Penn Tree Bank
# This file generates new sentences sampled from the language model
###############################################################################

import argparse
import torch

from utils import Index, SimpleRepl
from embedding import Embedding

parser = argparse.ArgumentParser(description='PyTorch Language Model')

# parameters
parser.add_argument('--index', type=str, default='../data/wikisentences/vocab.txt',
                    help='location of the vocabulary index')
parser.add_argument('--model', type=str, default='./model.pt',
                    help='model to use')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  if not args.cuda:
    print('WARNING: You have a CUDA device, so you should probably run with --cuda')

device = torch.device('cuda' if args.cuda else 'cpu')

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal to 1e-3")

def load(mfile, ifile):
  # load model
  print('Loading model', file=sys.stderr)
  with open(mfile, 'rb') as f:
    model = torch.load(f).to(device)
  model.rnn.flatten_parameters() # after load the rnn params are not a continuous chunk of memory. This makes them a continuous chunk, and will speed up forward pass
  model.eval() # deactivate training
  # load index
  print('Loading index', file=sys.stderr)
  index = Index.fromfile(ifile).freeze()
  print('Loading embedding', file=sys.stderr)
  emb = Embedding(model.encoder.weight.detach().numpy(), index, normalize = False)
  return model, index, emb

def generate(start = '<eos>', seqlen = 35, fout = sys.stdout):
  
  hidden = model.init_hidden(1)
  start = start.strip() if start is not None and start.strip() else '<eos>'
  if not index.hasWord(start):
    print("Word '%s' is unknown to the model." % start, file=sys.stderr)
    return
  sequence_input = torch.LongTensor([[ index[start] ]]).to(device)
  with torch.no_grad():  # no tracking history
    for i in range(seqlen):
      output, hidden = model(sequence_input, hidden)
      word_weights = output.squeeze().div(args.temperature).exp().cpu()
      word_idx = torch.multinomial(word_weights, 1)[0]
      sequence_input.fill_(word_idx)
      word = index.getWord(word_idx)
      print(word, file=fout, end=' ')
    print(file=fout)

model, index, embedding = load(args.model, args.index)

def nearest_neighbors(word = '<eos>', numneighbors = 10, fout = sys.stdout):
  word = word.strip() if word is not None and word.strip() else '<eos>'
  if not embedding.containsWord(word):
    print("Word '%s' is unknown to the model." % word, file=sys.stderr)
    return
  
  v = embedding.getVector(word)
  idxs, dists = embedding.search(v, topk=numneighbors)
  idxs = idxs[0,:]
  dists = dists[0,:]
  
  for i in range(len(idxs)):
    print('{:2d}: {:8d} ({:5.3}) {:s}'.format(
        i, 
        idxs[i], 
        dists[i], 
        index[idxs[i]]))  
    
def save_model(fname, tocpu=True, onnxformat=False):
  if tocpu:
    model_ = model.cpu()
  
  if not onnxformat:
    with open(fname, 'wb') as f:
      torch.save(model_, f)
    return

  print('The model is beeing exported in ONNX format at {}'.format(os.path.realpath(fname)))
  seqlen, batchsize = 35, 20
  dummy_input = torch.LongTensor(seqlen * batchsize).zero_().view(-1, batchsize)#.to(device)
  hidden = model.init_hidden(batchsize)
  torch.onnx.export(model, (dummy_input, hidden), fname)

def evalfun(cmd):
  commands = {
    'generate': lambda: 
      generate(
          start = input('Type start word: '), 
          seqlen = int(input('Type sequence length: '))
          ),
    'neighbors': lambda:
      nearest_neighbors(
          word = input('Type word: '), 
          numneighbors = int(input('Type number of nearest neighbors: '))
          ),
    'savemodel': lambda:
      save_model(
          file = input('Type filename: '), 
          tocpu = 'no' == str.lower(str.strip(input('Type no if the model should not be converted to cpu first.'))),
          onnxformat = 'yes' == str.lower(str.strip(input('Type yes if the model should be saved in onnx format.')))
          ),
    'help': lambda: 
      print('Type a valid command or CTRL-C to quit. \nValid commands: \n  ' + '\n  '.join(list(commands.keys()))) 
  }
  cmdfun = commands.get(cmd, commands['help'])
  try:
    cmdfun()
  except Exception as err:
    print(err, file=sys.stderr)
    print('Exception occurred!')
    commands['help']()

SimpleRepl(evalfun).run()
