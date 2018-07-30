# -*- coding: utf-8 -*-

# coding: utf-8

import sys
if not '..' in sys.path: sys.path.append('..')

import argparse
import time
import math
import os
import torch

from data import WikiSequence, CharSequence
from utils import Index, RandomBatchSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from embedding import Embedding, FastTextEmbedding, TextEmbedding, RandomEmbedding

from rnn_nets import RNNLM

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
parser.add_argument('--shuffle_batches', action='store_true',
                    help='shuffle batches')
parser.add_argument('--shuffle_samples', action='store_true',
                    help='shuffle samples')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
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

#index = Index()
#train_ = WikiSequence(args.data, subset='train', index = index, seqlen = args.bptt, skip = args.bptt).to(device)
#test_ = WikiSequence(args.data, subset='test', index = index, seqlen = args.bptt, skip = args.bptt).to(device)
#valid_ = WikiSequence(args.data, subset='valid', index = index, seqlen = args.bptt, skip = args.bptt).to(device)
#index.freeze().tofile(os.path.join(args.data, 'vocab.txt'))

train_ = CharSequence(filename='../data/tinyshakespeare.txt', seqlen=args.bptt, skip=args.bptt).to(device)
test_ = CharSequence(filename='../data/tinyshakespeare.txt', seqlen=args.bptt, skip=args.bptt).to(device)
valid_ = CharSequence(filename='../data/tinyshakespeare.txt', seqlen=args.bptt, skip=args.bptt).to(device)
index = train_.index


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

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def reshape_batch(batch_data):
    # dimensions: batch x seqlen
    x_batch, y_batch = batch_data
    # reshape x_batch so seqlen is dim 0 and batch is dim 1
    x_batch = x_batch.transpose(0,1) # switch dim 0 with dim 1
    # reshape y_batch so we get a 1d tensor of length seqlen x batch that matches with x_batch
    y_batch = y_batch.transpose(0,1).contiguous().view(-1) # switch dim 0 with dim 1 and view as 1d    
      #####
#        print(x_batch.size())
#        print(y_batch.size())
#        print('--- x')
#        print(list(map(list, index[x_batch.tolist()])))
#        print('--- y')
#        print(list(index[y_batch.tolist()]))
      #####
    return x_batch, y_batch


def evaluate(d_loader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(index)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for batch, batch_data in enumerate(d_loader):
            data, targets = reshape_batch(batch_data)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            loss_ = criterion(output_flat, targets)
            total_loss += len(data) * loss_.item()
            hidden = repackage_hidden(hidden)
            print('===i===\n', batch)
            print('===len(data)===\n', len(data))
            print('===loss===\n', loss_)
            print('===total loss===\n', total_loss)            
    print('===total loss===\n', total_loss)
    print('===len(d_loader)===\n',len(d_loader))
    print('===args.bptt===\n',args.bptt)
    return total_loss / (len(d_loader) * args.bptt )


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(index)
    hidden = model.init_hidden(args.batch_size)
    
    for batch, batch_data in enumerate(train_loader):
    
        data, targets = reshape_batch(batch_data)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, 
                batch, 
                len(train_loader), 
                lr,
                elapsed * 1000 / args.log_interval, 
                cur_loss, 
                math.exp(cur_loss)
                ))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(valid_loader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            epoch, 
            (time.time() - epoch_start_time), 
            val_loss, 
            math.exp(val_loss)
            ))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
#test_loss = evaluate(test_data)
test_loss = evaluate(test_loader)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

