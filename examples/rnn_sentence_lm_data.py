# -*- coding: utf-8 -*-

import os
import torch
import torch.utils.data
import sys

class Index(object):
  
  def __init__(self):
    self.id2w = []
    self.w2id = {}

  def add(self, word):
    if word in self.w2id:
      return self.w2id[word]
    idx = len(self.id2w)
    self.w2id[word] = idx
    self.id2w.append(word)
    return idx
  
  def size(self):
    return len(self.w2id)

  def getWord(self, index):
    return self.id2w[index]
    
  def getId(self, word):
    return self.w2id[word]
  
  def __len__(self):
    return self.size()

'''

'''
class WikiSentences(torch.utils.data.Dataset):

  def addword(self, word):
    idx = self.index.add(word)    
    self.data.append(idx)
    
  def tokenize(self, line):
    words = line.split() + ['<eos>']
    for w in words:
      self.addword(w)
  
  def load(self):
    print('Loading %s sentences from %s' % (self.subset, self.file), file=sys.stderr)
    self.data = []    
    assert os.path.exists(self.file)    
    with open(self.file, 'r', encoding='utf8') as f:
      for i, line in enumerate(f):
        self.tokenize(line)
    self.data = torch.LongTensor(self.data)

  def __init__(self, path, subset = 'train', index = None):
    super(WikiSentences, self)
    self.path = path
    self.subset = subset
    self.file = os.path.join(self.path, self.subset + '.txt')
    self.index = index if index is not None else Index()
    self.load()
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    self.data[index]
