#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:35:21 2018

@author: rem
"""
import sys
if not '..' in sys.path: sys.path.append('..')

import torch
import numpy as np
from embedding import FastTextEmbedding


#emb = TextEmbedding('/Users/rem/data/w2v/GoogleNews-vectors-negative300.txt').load(nlines=10000)
#emb = TextEmbedding('./GoogleNews-vectors-negative300.txt').load(nlines = 1000)
emb = FastTextEmbedding('../data/wiki.simple.bin').load()


def create_embedding_weights(emb, vocab = None):
  if not vocab:
    vocab = emb.vocabulary()
  weights = torch.zeros(len(vocab), emb.dim(), dtype = torch.float32 )
#  weights = torch.Tensor(np.zeros((len(vocab), emb.dim()), dtype = np.float32))
  for i, word in enumerate(vocab):
    if emb.containsWord(word):
      weights[i, :] = torch.Tensor(emb.getVector(word))
    else:
      weights[i, :] = torch.Tensor(np.random.normal(scale=0.6, size=(emb.dim(), )))
  return weights

def create_embedding_layer(weights, trainable = False):
  num_embeddings, embedding_dim = weights.size()
#  emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
#  emb_layer.load_state_dict({'weight': weights})
#  if not trainable:
#      emb_layer.weight.requires_grad = False
  emb_layer = torch.nn.Embedding.from_pretrained(weights, freeze = not trainable)
  return emb_layer, num_embeddings, embedding_dim



class NGramLanguageModeler(torch.nn.Module):

  def __init__(self, weights, context_size):
    super(NGramLanguageModeler, self).__init__()
    self.embeddings, num_embeddings, embedding_dim = create_embedding_layer(weights, trainable = True)
    self.linear1 = torch.nn.Linear(context_size * embedding_dim, 128)
    # num_embeddings == vocab_size
    self.linear2 = torch.nn.Linear(128, num_embeddings)

  def forward(self, inputs):
    embeds = self.embeddings(inputs).view((1, -1))
    out = torch.nn.functional.relu(self.linear1(embeds))
    out = self.linear2(out)
    log_probs = torch.nn.functional.log_softmax(out, dim=1)
    return log_probs

CONTEXT_SIZE = 2
# We will use Shakespeare Sonnet 2
test_sentence = '''When forty winters shall besiege thy brow,
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
And see thy blood warm when thou feel'st it cold.'''.split()
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]  
vocab = set(test_sentence)
weights = create_embedding_weights(emb, vocab)
vocab = np.array(list(vocab))
word_to_ix = { w: i for (i, w) in enumerate(vocab) }

losses = []
loss_function = torch.nn.NLLLoss()
net = NGramLanguageModeler(weights, CONTEXT_SIZE)
parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.1)

#  criterion = torch.nn.CrossEntropyLoss()
#  optimizer = torch.optim.Adam(net.parameters())


for epoch in range(100):
  total_loss = 0
  for context, target in trigrams:

    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
    # into integer indices and wrap them in tensors)
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

    # Step 2. Recall that torch *accumulates* gradients. Before passing in a
    # new instance, you need to zero out the gradients from the old
    # instance
    net.zero_grad()

    # Step 3. Run the forward pass, getting log probabilities over next
    # words
    log_probs = net(context_idxs)

    # Step 4. Compute your loss function. (Again, Torch wants the target
    # word wrapped in a tensor)
    target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)
    loss = loss_function(log_probs, target_idx)

    # Step 5. Do the backward pass and update the gradient
    loss.backward()
    optimizer.step()

    # Get the Python number from a 1-element Tensor by calling tensor.item()
    total_loss += loss.item()
  losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!


# predict
context_idxs = torch.tensor([word_to_ix[w] for w in 'in thy'.split()], dtype=torch.long)
log_probs = net(inputs=context_idxs)
maxid = log_probs.argmax().item()
vals, ids = torch.sort(log_probs, descending = True)
print(vals[0,:5].exp().tolist())
print(vocab[ids[0,:5].tolist()])

  
  