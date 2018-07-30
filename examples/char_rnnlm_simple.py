#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:35:21 2018

@author: rem
"""
import sys
if not '..' in sys.path: sys.path.append('..')

import torch
from data import CharDataset
from rnn_nets import SimpleRNN
import time
import random

# We will use Shakespeare Sonnet 2
charsequence = '''When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.'''

# prepare data
# create a Tensor of long values representing indices of the chars
dataset = CharDataset(filename='../data/tinyshakespeare.txt', seqlen=100)

# prepare model
model = SimpleRNN(ntoken = len(dataset.index), emsize=300, nhid=200, nlayers=1) # ntokens = nchars
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# train
def get_random_sample():
  index = random.randint(0, len(dataset))
  return dataset[index]

def process(inputs, targets):
  hidden = model.init_hidden()
  model.zero_grad()
  loss = 0.

  # generate
  for i in range(len(inputs)):
    output, hidden = model(inputs[i:i+1], hidden)
    loss += criterion(output, targets[i:i+1])

  loss.backward()
  optimizer.step()
  
  return loss.data.item() / len(inputs)

def generate(prime='A', predict_len=100, temperature=0.8):
  prime = list(prime)
  hidden = model.init_hidden()
  prime_input = torch.LongTensor(list(dataset.index[prime]))
  predicted = prime

  # Use priming string to "build up" hidden state
  for i in range(len(prime) - 1):
    _, hidden = model(prime_input[i], hidden)
      
  inp = prime_input[-1]
  
  for p in range(predict_len):
    output, hidden = model(inp, hidden)
    
    # Sample from the network as a multinomial distribution
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1)[0]
    
    # Add predicted character to string and use as next input
    predicted_char = dataset.index[top_i.item()]
    predicted += predicted_char
    inp = torch.LongTensor([top_i])
  return ''.join(predicted)

start_time = time.time()
all_losses = []
loss_avg = 0
epochs = 2000

print("Training for %d epochs..." % epochs)
for epoch in range(epochs):
  inputs, targets = get_random_sample()

  loss = process(inputs, targets)
  loss_avg += loss

  if epoch % 200 == 0:
    print('[%s (%d %d%%) %.4f]' % (time.time() - start_time, epoch, epoch / epochs * 100, loss))
  if epoch % 400 == 0:    
    print(generate(prime='Wh', predict_len=20), '\n')
