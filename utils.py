#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:15:22 2018

@author: rem
"""

import random
import torch

class Index(object):
  
  def __init__(self):
    self.id2w = []
    self.w2id = {}
    self.frozen = False

  def add(self, word):
    if word in self.w2id:
      return self.w2id[word]
    if self.frozen:
      raise ValueError('Index can not be altered anymore. It is already frozen.')
    idx = len(self.id2w)
    self.w2id[word] = idx
    self.id2w.append(word)
    return idx
  
  def size(self):
    return len(self.w2id)
  
  def hasId(self, idx):
    return idx >= 0 and idx < len(self.id2w)
  
  def hasWord(self, word):
    return word in self.w2id

  def getWord(self, index):
    return self.id2w[index]
    
  def getId(self, word):
    return self.w2id[word]
  
  def tofile(self, fname):
    with open(fname, 'w') as f:
      lines = map(lambda w: w + '\n', self.id2w)
      f.writelines(lines)
      
  def freeze(self):
    self.frozen = True
    return self
  
  def vocabulary(self):
    return self.id2w
  
  def __contains__(self, key):
    if isinstance(key, str):
      return self.hasWord(key)
    return self.hasId(key)

  def __getitem__(self, key):
    # return the id if we get a word
    if isinstance(key, str):
      return self.getId(key)
    # return the word if we get an id, lets assume that 'key' is some kind of number, i.e. int, long, ...
    if not hasattr(key, '__iter__'):
      return self.getWord(key)
    # otherwise recursively apply this method for every key in an iterable
    return map(lambda k: self[k], key)
  
  def __iter__(self):
    return self.id2w.__iter__()

  def __len__(self):
    return self.size()
  
  @staticmethod
  def fromfile(fname):
    index = Index()
    with open(fname, 'r', encoding='utf8') as f:
      for i, line in enumerate(f):
        w = line.rstrip()
        index.id2w.append(w)
        index.w2id[w] = i
    return index

class RandomBatchSampler(torch.utils.data.sampler.BatchSampler):
  
  def __init__(self, *args, **kwargs):
    super(RandomBatchSampler, self).__init__(*args, **kwargs)
    
  def __iter__(self):
    batches = list(super().__iter__())
    random.shuffle(batches)
    for batch in batches:
      yield batch