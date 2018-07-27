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
from index import Index
from embedding import Embedding, FastTextEmbedding

import rnn_sentence_lm_net as model

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
preemb = FastTextEmbedding('../data/wiki.simple.bin').load()
preemb = Embedding.filterembedding(index.vocabulary(), preemb, fillmissing = False)
preemb_weights = preemb.weights

eval_batch_size = 10
train_loader = torch.utils.data.DataLoader(train_, batch_size = args.batch_size, drop_last = True, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_, batch_size = eval_batch_size, drop_last = True, num_workers = 0)
valid_loader = torch.utils.data.DataLoader(valid_, batch_size = eval_batch_size, drop_last = True, num_workers = 0)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

###############################################################################
# Build the model
###############################################################################

ntokens = len(index)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, init_weights = preemb_weights).to(device)

criterion = nn.CrossEntropyLoss()

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
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
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

##########
##########
#        print('--- data')
#        print(data.size())
#        print(list(map(list, index[data.tolist()])))
#        print('--- targets')
#        print(targets.size())
#        print(list(index[targets.tolist()]))
#        print('--- output')
#        print(output.size())
#        output_squeezed = output.view(-1, ntokens)
#        print('--- output view')
#        print(output_squeezed.size())
#        print(output_squeezed)
#        sys.exit()
##########
##########
        
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
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


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


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

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
