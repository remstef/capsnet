#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:32:26 2018

@author: rem
"""

import numpy as np
import os
import pandas
import csv
import torch
import torch.utils
import torch.utils.data
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from embedding import Embedding, RandomEmbedding, TextEmbedding, FastTextEmbedding
from index import Index


from sklearn.datasets import fetch_20newsgroups
import pickle

emb = RandomEmbedding(300) # random vectors
#emb = TextEmbedding('./GoogleNews-vectors-negative300.txt').load(nlines = 1000)
#emb = TextEmbedding('./glove.840B.300d.txt').load(nlines = 1000, skipheader = False, normalize = True, nlines = 1000)
#emb = FastTextEmbedding('./wiki.en.bin').load()

  
'''
 Sentences as a sequence in a single 1d tensor
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
    
  def __init__(self, path, subset = 'train', index = None, seqlen = 35):
    super(WikiSentences, self)
    self.path = path
    self.subset = subset
    self.file = os.path.join(self.path, self.subset + '.txt')
    self.index = index if index is not None else Index()
    self.load()
    self.seqlen = seqlen
    self.sequences = len(self.data) // self.seqlen - self.seqlen
    
  def __len__(self):
    return self.sequences - 1

  def __getitem__(self, index):
    # get the sequence for index 
    skip_index = index * self.seqlen # make sure each sequence is only read once
    x = self.data[skip_index     : skip_index + self.seqlen    ]
    y = self.data[skip_index + 1 : skip_index + self.seqlen + 1]
    return x, y 
  
  def cuda(self):
    self.data = self.data.cuda()
    return self
  
  def to(self, device):
    self.data = self.data.to(device)
    return self

class SpamDataset(torch.utils.data.Dataset):

    '''
    maxlength = size of padding
    '''
    def __init__(self, maxlength = 30, subset='all'):
      super(SpamDataset, self)
      self.maxlength = maxlength
      self.subset = subset
      self.samples, self.labels = self.load_data()
      self.samples_vectors = self.load_samples_vectors()

    def load_data(self):
      # do some preprocessing if preprocessed file does not exist
      if not os.path.isfile('SMSSpamCollection_normalized'):
        #import spacy; nlp=spacy.load('en')
        print('Applying spacy.')
        import en_core_web_sm
        nlp = en_core_web_sm.load()

        def normalize(message):
          doc = nlp(message)
          normalized = doc
          normalized = filter(lambda t : t.is_alpha and not t.is_stop, doc)
          normalized = map(lambda t : t.text, normalized)
          normalized = list(normalized)
          return normalized

        messages = pandas.read_csv(
            'SMSSpamCollection',
            sep='\t',
            quoting=csv.QUOTE_NONE,
            names=['label', 'message'],
            encoding='UTF-8')
        messages['normalized'] = messages.message.apply(normalize)
        messages.to_pickle('SMSSpamCollection_normalized')

      # load normalized messages
      messages = pandas.read_pickle('SMSSpamCollection_normalized')
      samples = messages.normalized.tolist()
      labels = messages.label.tolist()

      self.classes = list(set(labels))
      self.classes.sort()   #Otherwise it is inconsistent which class is 0 and which is 1
      self.oh_classes = MultiLabelBinarizer()
      self.oh_classes.fit([[label] for label in self.classes])
      self.classes = dict([(l,i) for (i,l) in enumerate(self.oh_classes.classes_)])

      labels = [self.classes[l] for l in labels]
      cut_off = int(len(samples) * 0.2)
      if self.subset == 'train':
        samples = samples[cut_off:]
        labels =labels[cut_off:]
      elif self.subset == 'test':
        samples= samples[:cut_off]
        labels = labels[:cut_off]
      return samples, labels

    def __len__(self):
      return len(self.samples)

    """Returns (item as matrix of token vectors, label, index) instead of the normal (item, label). See constructor for more info"""
    def __getitem__(self, index):
      msg = self.samples[index]
      int_label = self.classes[self.labels[index]]   #Convert label to integer. Is this slow?
      #mat = np.zeros( (self.maxlength, self.ftmodel.numpy_normalized_vectors.shape[1]) )
      mat = np.zeros( (self.maxlength, 300) )   #Hard coded for the wikipedia data that has dim=300

      for i, token in enumerate(msg):
        if i >= self.maxlength:
          break
        v = emb.getVector(token)
        mat[i,:] = v
        oh_label = self.oh_classes.transform([[self.labels[index]]])
        oh_label = oh_label[0]
        #print(oh_label)

      return ( torch.tensor(mat), int_label, oh_label, index )

    def get_sample(self, index):
        return self.samples[index]


    def load_samples_vectors(self):
        #Average Word_count train: 94
        #Average Word_count test: 88.2
        # av_len = 0
        # for msg in self.samples:
        #     av_len+= len(msg)
        # print(av_len / len(self.samples))
        #print(len(self.samples))

        mat = np.zeros( (len(self.samples),self.maxlength, 300) )
        # filter unknowns
        for j, msg in enumerate(self.samples):
          for i, token in enumerate(msg):
            word = self.samples[j][i]
            if not emb.containsWord(word):
              self.samples[j].remove(word)
            if emb.containsWord(word) and not emb.containsWord(token):
              print(word, token, " DAFUQ")

        for j, msg in enumerate(self.samples):
          #print(msg)
          for i, token in enumerate(msg):
            if i >= self.maxlength:
              break
            if not emb.containsWord(token):
              #print(token)
              continue
            v = emb.getVector(token)
            #print(v)
            mat[j, i, :] = v
        return mat


class NewsGroupDataset(torch.utils.data.Dataset):

    '''
    maxlength = size of padding
    '''
    def __init__(self, subset='all',  maxlength = 30):
      super(NewsGroupDataset, self)
      self.maxlength = maxlength
      self.subset = subset
      self.file_name = "20_newsgroup_normalized_" + subset
      self.samples, self.labels = self.load_data()
      self.samples_vectors = self.load_samples_vectors()


    def load_data(self):
      messages = fetch_20newsgroups(subset=self.subset, remove=('headers','footers','quotes'), shuffle=True, random_state=42)
      # do some preprocessing if preprocessed file does not exist
      if not os.path.isfile(self.file_name):
        #import spacy; nlp=spacy.load('en')
        print('Applying spacy.')
        import en_core_web_sm
        nlp = en_core_web_sm.load()

        def normalize(message):
          doc = nlp(message)
          normalized = doc
          normalized = filter(lambda t : t.is_alpha and not t.is_stop, doc)
          normalized = map(lambda t : t.text, normalized)
          normalized = list(normalized)
          return normalized

        messages_normalized = list(map(normalize, messages.data))
        with open(self.file_name, 'wb') as f:
          pickle.dump(messages_normalized, f)

      with open(self.file_name, 'rb') as f:
        messages_normalized = pickle.load(f)

      samples = messages_normalized
      labels = messages.target

      self.classes = list(set(labels))
      self.classes.sort()   #Otherwise it is inconsistent which class is 0 and which is 1
      self.oh_classes = MultiLabelBinarizer()
      self.oh_classes.fit([[label] for label in self.classes])
      self.classes = dict([(l,i) for (i,l) in enumerate(self.oh_classes.classes_)])

      return samples, labels

    def __len__(self):
      return len(self.samples)

    """Returns (item as matrix of token vectors, label, index) instead of the normal (item, label). See constructor for more info"""
    def __getitem__(self, index):
      msg = self.samples[index]
      label = self.classes[self.labels[index]]   #Convert label to integer. Is this slow?
      oh_label = self.oh_classes.transform([[self.labels[index]]])
      oh_label = oh_label[0]
      #mat = np.zeros( (self.maxlength, self.ftmodel.numpy_normalized_vectors.shape[1]) )
      mat = np.zeros( (self.maxlength, 300) )   #Hard coded for the wikipedia data that has dim=300


      for i, token in enumerate(msg):
        if i >= self.maxlength:
          break
        if not emb.containsWord(token):
          continue
        v = emb.getVector(token)
        mat[i,:] = v

      return ( torch.tensor(mat), label, oh_label,  index )

    def get_sample(self, index):
        return self.samples[index]

    def load_samples_vectors(self):
        #Average Word_count train: 94
        #Average Word_count test: 88.2
        # av_len = 0
        # for msg in self.samples:
        #     av_len+= len(msg)
        # print(av_len / len(self.samples))
        #print(len(self.samples))
        mat = np.zeros( (len(self.samples),self.maxlength, 300) )

        # filter unknowns
        for j, msg in enumerate(self.samples):
          for i, token in enumerate(msg):
            word = self.samples[j][i]
            if not emb.containsWord(word):
              self.samples[j].remove(word)
            if emb.containsWord(word) and not emb.containsWord(token):
              print(word, token, " DAFUQ")

        for j, msg in enumerate(self.samples):
          #print(msg)
          for i, token in enumerate(msg):
            if i >= self.maxlength:
              break
            if not emb.containsWord(token):
              #print(token)
              continue
            v = emb.getVector(token)
            #print(v)
            mat[j, i, :] = v
        return mat

        return mat

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes, dimensions, num_samples):
        print("WARNING: USING A DUMMY DATASET")
        self.classes = range(num_classes)
        self.dimensions = dimensions
        self.num_samples = num_samples
        self.classes = [str(i) for i in range(num_classes)]
        self.max_length = dimensions[0]
        self.embeddings = None
        self.embedding_dim = dimensions[1]

    def __len__(self):
        print (self.num_samples)
        return self.num_samples

    def __getitem__(self, index):
        cycle = index % len(self.classes)
        sample = torch.zeros(self.dimensions) + cycle * (1/len(self.classes))
        label = cycle
        label_oh = torch.zeros(len(self.classes))
        label_oh[cycle] = 1
        return sample, label, label_oh, index

    def get_data_dimensions(self):
        return self.dimensions
      
class SemEval2008(torch.utils.data.Dataset):

    '''
      maxlength = size of padding
    '''
    def __init__(self, maxlength = 30):
      super(SemEval2008, self)
      self.load_data()

    def load_data(self):
      # do some preprocessing if preprocessed file does not exist
      if not os.path.isfile('semeval-2008-sample-processed.txt'):
        #import spacy; nlp=spacy.load('en')
        print('Applying spacy.')
        import en_core_web_sm
        nlp = en_core_web_sm.load()

        samples = pandas.read_csv(
            'semeval-2008-sample.txt',
            sep=' ',
            quoting=csv.QUOTE_ALL,
            names=['id', 'originalsentence'],
            encoding='UTF-8')
        samples['original_offset_e1'] = samples.originalsentence.apply(lambda s: (s.find('<e1>'), s.find('</e1>')))
        samples['original_offset_e2'] = samples.originalsentence.apply(lambda s: (s.find('<e2>'), s.find('</e2>')))
        samples['e1'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e1[0] + 4:row.original_offset_e1[1]], axis=1)
        samples['e2'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e2[0] + 4:row.original_offset_e2[1]], axis=1)
        samples['l'] = samples.apply(lambda row: row.originalsentence[:row.original_offset_e1[0]], axis=1)
        samples['r'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e2[1]+5:], axis=1)
        samples['m'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e1[1]+5:row.original_offset_e2[0]], axis=1)
        samples['sentence'] = samples.apply(lambda row: row.l+row.e1+row.m+row.e2+row.r, axis=1)
        samples['offset_e1'] = samples.apply(lambda row: (len(row.l), len(row.l) + len(row.e1)), axis=1)
        samples['offset_e2'] = samples.apply(lambda row: (len(row.l) + len(row.e1) + len(row.m), len(row.l) + len(row.e1) + len(row.m) + len(row.e2)), axis=1)
        samples['spacy'] = samples.sentence.apply(lambda s: nlp(s))
        samples.to_pickle('semeval-2008-sample-processed.pickle')
        del samples

      # load processed messages
      self.samples = pandas.read_pickle('semeval-2008-sample-processed.pickle')

      self.classes = list(set(samples.labels))
      self.classes.sort()
      self.classes = dict([(l,i) for (i,l) in enumerate(self.classes)])
      
      return True
    

    def __len__(self):
      return len(self.samples)


    '''
      @return (item as matrix of token vectors, label, index) instead of the normal (item, label)
    '''
    def __getitem__(self, index):
      msg = self.samples[index]
      int_label = self.classes[self.labels[index]]   #Convert label to integer. Is this slow?
      #mat = np.zeros( (self.maxlength, self.ftmodel.numpy_normalized_vectors.shape[1]) )
      mat = np.zeros( (self.maxlength, 300) )   #Hard coded for the wikipedia data that has dim=300

      for i, token in enumerate(msg):

        if i >= self.maxlength:
          break
        if not emb.containsWord(token):
          continue
        v = emb.getVector(token)

        mat[i,:] = v
        oh_label = self.oh_classes.transform([[self.labels[index]]])
        oh_label = oh_label[0]
        #print(oh_label)

      return ( torch.tensor(mat), int_label, oh_label, index )

    def get_sample(self, index):
        return self.samples[index]

