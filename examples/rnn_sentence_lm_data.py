# -*- coding: utf-8 -*-

import os
import torch
import torch.utils.data
import sys

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
  
  @staticmethod
  def fromfile(fname):
    index = Index()
    with open(fname, 'r', encoding='utf8') as f:
      for i, line in enumerate(f):
        w = line.rstrip()
        index.id2w.append(w)
        index.w2id[w] = i
    return index
  
  def __getitem__(self, key):
    if isinstance(key, str):
      return self.getId(key)
    if hasattr(key, '__iter__'):
      return list(map(lambda k: self[k], key))
    else:
      return self.getWord(key)
  
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
