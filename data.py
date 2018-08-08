#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:32:26 2018

@author: rem
"""

import numpy as np
import math
import os
import pandas
import csv
import torch
from tqdm import tqdm
import torch.utils
import torch.utils.data
import sys
from sklearn.preprocessing import MultiLabelBinarizer
#from embedding import Embedding, RandomEmbedding, TextEmbedding, FastTextEmbedding
from utils import Index


from sklearn.datasets import fetch_20newsgroups
import pickle

'''
 Sequences in a single 1d tensor
'''
class FixedLengthSequenceDataset(torch.utils.data.Dataset):

  def __init__(self, seqlen = 35, skip = 35):
    super(FixedLengthSequenceDataset, self)
    self.seqlen = seqlen
    self.skip = skip
    self.data = None
    self.nsequences = None
    
  def __len__(self):
    if self.nsequences is None:
      self.nsequences = int(math.ceil((len(self.data)-self.seqlen) / self.skip))
    return self.nsequences
    
  def __getitem__(self, index):
    # seqlen=4
    # skip=3
    #    abcdefghijkl
    # x1 ----         
    # y1  ----         
    # x2    ----       
    # y2     ----       
    # x3       ----      
    # y3        ----      
    # x4          xxx
    # y4           xx
    #
    # seqlen=4
    # skip=2
    #    abcdefghijkl
    # x1 ----         
    # y1  ----         
    # x2   ----       
    # y2    ----       
    # x3     ----      
    # y3      ----      
    # x4       ----  
    # y4        ----  
    # x5         ----
    # y5          xxx
    # get the sequence for index 
    skip_index = index * self.skip # make sure each sequence is only read once
    x = self.data[skip_index     : skip_index + self.seqlen    ]
    y = self.data[skip_index + 1 : skip_index + self.seqlen + 1]
    return x, y, self.seqlen
  
  def cuda(self):
    self.data = self.data.cuda()
    return self
  
  def to(self, device):
    self.data = self.data.to(device)
    return self
  

class CharSequence(FixedLengthSequenceDataset):
  
  def load(self):
    print('Loading chars from %s' % self.file, file=sys.stderr)    
    assert os.path.exists(self.file)    
    with open(self.file, 'r', encoding='utf8') as f:
      charsequence = f.read()
    sequence = torch.LongTensor(list(map(lambda c: self.index.add(c.strip()), charsequence)))
    del charsequence
    return sequence
    
  def __init__(self, path, subset = 'train.txt', index = None, seqlen = 35, skip = 35):
    super(CharSequence, self).__init__(seqlen, skip)
    self.path = path
    self.subset = subset
    self.file = os.path.join(self.path, self.subset)
    self.index = index if index is not None else Index()
    self.data = self.load()
    
  
'''
 tokens as a sequence in a single 1d tensor
'''
class TokenSequence(FixedLengthSequenceDataset):
  
  def load(self):
    print('Loading %s sentences from %s' % (self.subset, self.file), file=sys.stderr)    
    assert os.path.exists(self.file)    
    data_ = []
    with open(self.file, 'r', encoding='utf8') as f:
      for i, line in enumerate(f):
        for w in line.split() + ['<eos>']:
          data_.append(self.index.add(w.strip()))
    data = torch.LongTensor(data_)
    del data_
    return data
    
  def __init__(self, path, subset = 'train.txt', index = None, seqlen = 35, skip = 35):
    super(TokenSequence, self).__init__(seqlen, skip)
    self.path = path
    self.subset = subset
    self.file = os.path.join(self.path, self.subset)
    self.index = index if index is not None else Index()
    self.data = self.load()
    
'''

'''
class SemEval2010(torch.utils.data.Dataset):

  def __init__(self, path, subset = 'train.txt', index = None, classindex = None):
    self.path = path
    self.subset = subset
    self.index = index if index is not None else Index()
    self.classindex = classindex if classindex is not None else Index()
    self.load()

  def load(self):
    source_file = os.path.join(self.path, self.subset)
    processed_file = source_file + '.pkl'
    
    # do some preprocessing if preprocessed file does not exist
    if not os.path.isfile(processed_file):
      #import spacy; nlp=spacy.load('en')
      print('Applying spacy.')
      import en_core_web_sm
      nlp = en_core_web_sm.load()
      tqdm.pandas()
      samples = pandas.read_csv(
          source_file,
          sep='\t',
          delim_whitespace = False,
          quoting=csv.QUOTE_MINIMAL,
          names=['id', 'originalsentence', 'label', 'comment'],
          skip_blank_lines=True,
          encoding='utf-8')
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
      samples['spacy'] = samples.sentence.progress_apply(lambda s: nlp(s))
      samples.to_pickle(processed_file)
      del samples

    # load processed messages
    self.samples = pandas.read_pickle(processed_file)
    self.samples['sentence_tensor'] = self.samples.spacy.apply(lambda doc: torch.LongTensor([self.index.add(t.text.strip()) for t in doc]))
    self.samples['sentence_tensor'] = self.samples.sentence_tensor.apply(lambda t: torch.cat((t,torch.LongTensor([self.index.add('<eos>')])),0))    
    self.samples['sentence_length'] = self.samples.sentence_tensor.apply(lambda t: t.size(0))
    self.maxseqlen = self.samples.sentence_length.max()
    pad_val = self.index.add('<pad>')
    self.samples['sentence_tensor_padded'] = self.samples.sentence_tensor.apply(lambda t: self.pad(t, self.maxseqlen, pad_val))
    self.samples['labelid'] = self.samples.label.apply(lambda lbl: self.classindex.add(lbl.strip()))

    self.sequences = torch.stack(self.samples.sentence_tensor_padded.tolist())
    self.sequencelengts = torch.LongTensor(self.samples.sentence_length.tolist())
    self.labels = torch.LongTensor(self.samples.labelid.tolist())
    
    del self.samples
    
    return True
  
  def __len__(self):
    return self.labels.size(0)

  def __getitem__(self, index):
    x = self.sequences[index]
    l = self.sequencelengts[index]
    y = self.labels[index]
    return x, y, l
  
  def pad(self, x, length, padding_value):
    y = torch.ones((length,)).long() * padding_value
    y[:len(x)] = x
    return y

  def cpu(self):
    return self.to(torch.device('cpu'))
  
  def cuda(self):
    return self.to(torch.device('cuda'))
  
  def to(self, device):
    self.sequences = self.sequences.to(device)
    self.sequencelengts = self.sequencelengts.to(device)
    self.labels = self.labels.to(device)
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
      if not os.path.isfile('SMSSpamCollection_normalized.pkl'):
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

#    def get_sample(self, index):
#        return self.samples[index]


#    def load_samples_vectors(self):
#        #Average Word_count train: 94
#        #Average Word_count test: 88.2
#        # av_len = 0
#        # for msg in self.samples:
#        #     av_len+= len(msg)
#        # print(av_len / len(self.samples))
#        #print(len(self.samples))
#
#        mat = np.zeros( (len(self.samples),self.maxlength, 300) )
#        # filter unknowns
#        for j, msg in enumerate(self.samples):
#          for i, token in enumerate(msg):
#            word = self.samples[j][i]
#            if not emb.containsWord(word):
#              self.samples[j].remove(word)
#            if emb.containsWord(word) and not emb.containsWord(token):
#              print(word, token, " DAFUQ")
#
#        for j, msg in enumerate(self.samples):
#          #print(msg)
#          for i, token in enumerate(msg):
#            if i >= self.maxlength:
#              break
#            if not emb.containsWord(token):
#              #print(token)
#              continue
#            v = emb.getVector(token)
#            #print(v)
#            mat[j, i, :] = v
#        return mat


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

#    def load_samples_vectors(self):
#        #Average Word_count train: 94
#        #Average Word_count test: 88.2
#        # av_len = 0
#        # for msg in self.samples:
#        #     av_len+= len(msg)
#        # print(av_len / len(self.samples))
#        #print(len(self.samples))
#        mat = np.zeros( (len(self.samples),self.maxlength, 300) )
#
#        # filter unknowns
#        for j, msg in enumerate(self.samples):
#          for i, token in enumerate(msg):
#            word = self.samples[j][i]
#            if not emb.containsWord(word):
#              self.samples[j].remove(word)
#            if emb.containsWord(word) and not emb.containsWord(token):
#              print(word, token, " DAFUQ")
#
#        for j, msg in enumerate(self.samples):
#          #print(msg)
#          for i, token in enumerate(msg):
#            if i >= self.maxlength:
#              break
#            if not emb.containsWord(token):
#              #print(token)
#              continue
#            v = emb.getVector(token)
#            #print(v)
#            mat[j, i, :] = v
#        return mat


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


