#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:42:50 2018

@author: rem
"""

import numpy as np
import sys
import faiss
from pyfasttext import FastText
from utils import Index

class Embedding(object):
  
  def __init__(self, weights, index):
    assert weights.shape[0] == len(index)
    self.vdim = weights.shape[1]
    self.index = index
    self.weights = weights
    self.invindex = None

  def getVector(self, word):
    if not self.containsWord(word):
      print("'%s' is unknown." % word, file = sys.stderr)
      return np.zeros(self.vdim)
    idx = self.index.getId(word)
    return self.weights[idx]
    
  def search(self, q, topk = 4):
    if not self.invindex:
      print('Building faiss index...')
      self.invindex = faiss.IndexFlatL2(self.vdim)
      self.invindex.add(self.weights)
      print('Faiss index built:', self.invindex.is_trained)
    if len(q.shape) == 1:
      q = np.matrix(q)
    if q.shape[1] != self.vdim:
      print('Wrong shape, expected %d dimensions but got %d.' % (self.vdim, q.shape[1]), file = sys.stderr)
      return
    D, I = self.invindex.search(q, topk) # D = distances, I = indices
    return ( I, D )
    
  def wordForVec(self, v):
    idx, dist = self.search(v, topk=1)
    idx = idx[0,0]
    dist = dist[0,0]
    sim = 1. - dist
    word = self.index.getWord[idx]
    return word, sim

  def containsId(self, idx):
    return self.index.hasId(idx)
  
  def containsWord(self, word):
    return self.index.hasWord(word)
  
  def vocabulary(self):
    return self.id2w
  
  def dim(self):
    return self.vdim
  
  @staticmethod
  def filteredEmbedding(vocabulary, embedding, fillmissing = True):
    index = Index()
    weights = []
    if fillmissing:
      rv = RandomEmbedding(embedding.dim())
    for w in vocabulary:
      if index.hasWord(w):
        continue
      if embedding.containsWord(w):
        index.add(w)
        weights.append(embedding.getVector(w))
      elif fillmissing:
        index.add(w)
        weights.append(rv.getVector(w))
    weights = np.array(weights, dtype = np.float32)
    return Embedding(weights, index)
  
  
class RandomEmbedding(Embedding):
  
  def __init__(self, vectordim = 300):
    self.index = Index()
    self.vdim = vectordim
    self.data = np.zeros((0, self.vdim), dtype = np.float32)
    self.invindex = None
  
  def getVector(self, word):
    if not self.index.hasWord(word):
      # create random vector
      v = np.random.rand(self.vdim).astype(np.float32)
      # normalize
      length = np.linalg.norm(v)
      if length == 0:
        length += 1e-6
      v = v / length
      # add
      idx = self.index.add(self.id2w)
      self.data = np.vstack((self.data, v))
      assert idx == len(self.data)
      if self.invindex is not None:
        del self.invindex
        self.invindex = None
      return v
    idx = self.index.getId(word)
    return self.data[idx]
    
  def search(self, q, topk = 4):
    if not self.invindex:
      print('Building faiss index...')
      self.invindex = faiss.IndexFlatL2(self.vdim)
      self.invindex.add(self.data)
      print('Faiss index built:', self.invindex.is_trained)
    if len(q.shape) == 1:
      q = np.matrix(q)
    if q.shape[1] != self.vdim:
      print('Wrong shape, expected %d dimensions but got %d.' % (self.vdim, q.shape[1]), file = sys.stderr)
      return
    D, I = self.invindex.search(q, topk) # D = distances, I = indices
    return ( I, D )
    
  def wordForVec(self, v):
    idx, dist = self.search(v, topk=1)
    idx = idx[0,0]
    dist = dist[0,0]
    sim = 1. - dist
    word = self.index.getWord(idx)
    return word, sim
  
  def containsWord(self, word):
    return True
  
  def vocabulary(self):
    return self.index.vocbulary()
  
  def dim(self):
    return self.vdim

    
class FastTextEmbedding(Embedding):

  def __init__(self, binfile, normalize = False):
    self.file = binfile
    self.vdim = -1
    self.normalize = normalize
    
  def load(self):
    print('Loading fasttext model.')
    self.ftmodel = FastText()
    self.ftmodel.load_model(self.file)
    self.vdim = len(self.ftmodel['is'])
    print('Finished loading fasttext model.')
    return self
  
  def getVector(self, word):
    return self.ftmodel.get_numpy_vector(word, normalized = self.normalize)
    
  def search(self, q, topk = 4):
    raise NotImplementedError()
    
  def wordForVec(self, v):
    word, sim = self.ftmodel.words_for_vector(v)[0]
    return word, sim
  
  def containsWord(self, word):
    return True
  
  def vocabulary(self):
    return self.ftmodel.words
  
  def dim(self):
    return self.vdim
    

class TextEmbedding(Embedding):
  
  def __init__(self, txtfile, sep = ' ', vectordim = 300):
    self.file = txtfile
    self.vdim = vectordim
    self.separator = sep
    
  def load(self, skipheader = True, nlines = sys.maxsize, normalize = False):
    self.index = Index()
    print('Loading embedding from %s' % self.file)
    data_ = []
    with open(self.file, 'r', encoding='utf-8', errors='ignore') as f:
      if skipheader:
        f.readline()
      for i, line in enumerate(f):
        if i >= nlines:
          break
        try:
          line = line.strip()
          splits = line.split(self.separator)
          word = splits[0]
          if self.index.hasWord(word):
            continue
          coefs = np.array(splits[1:self.vdim+1], dtype=np.float32)
          if normalize:
            length = np.linalg.norm(coefs)
            if length == 0:
              length += 1e-6
            coefs = coefs / length
          if coefs.shape != (self.vdim,):
            continue
          idx = self.index.add(word)
          data_.append(coefs)
          assert idx == len(data_)
        except Exception as err:
          print('Error in line %d' % i, sys.exc_info()[0], file = sys.stderr)
          print(' ', err, file = sys.stderr)
          continue
    self.data = np.array(data_, dtype = np.float32)
    del data_
    print('Building faiss index...')
    if not self.normalize:
      print('Attention, normlization of vectors is required to guarantee functional search behaviour. Be sure your vectors are normalized, otherwise declare normlaize flag!')
    self.invindex = faiss.IndexFlatL2(self.vdim)
    self.invindex.add(self.data)
    print('Faiss index built:', self.invindex.is_trained)
    return self
  
  def getVector(self, word):
    if not self.containsWord(word):
      print("'%s' is unknown." % word, file = sys.stderr)
      v = np.zeros(self.vdim)
      v[0] = 1
      return v
    idx = self.index.getId(word)
    return self.data[idx]
    
  def search(self, q, topk = 4):
    if len(q.shape) == 1:
      q = np.matrix(q)
    if q.shape[1] != self.vdim:
      print('Wrong shape, expected %d dimensions but got %d.' % (self.vdim, q.shape[1]), file = sys.stderr )
      return
    D, I = self.invindex.search(q, topk) # D = distances, I = indices
    return ( I, D )
    
  def wordForVec(self, v):
    idx, dist = self.search(v, topk=1)
    idx = idx[0,0]
    dist = dist[0,0]
    sim = 1. - dist
    word = self.index.getWord(idx)
    return word, sim
  
  def containsWord(self, word):
    return self.index.hasWord(word)
  
  def vocabulary(self):
    return self.index.vocabulary()
  
  def dim(self):
    return self.vdim
  
    
#words = []
#idx = 0
#word2idx = {}
#vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')
#
#with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
#    for l in f:
#        line = l.decode().split()
#        word = line[0]
#        words.append(word)
#        word2idx[word] = idx
#        idx += 1
#        vect = np.array(line[1:]).astype(np.float)
#        vectors.append(vect)
#    
#vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
#vectors.flush()
#pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
#pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))
  
  
#vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
#words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
#word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
#glove = {w: vectors[word2idx[w]] for w in words}

    
