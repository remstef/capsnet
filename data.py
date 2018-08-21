#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: rem
'''
import sys
import os
import csv
import math
import re
import numpy as np
import pandas
import torch
from tqdm import tqdm
import torch.utils
import torch.utils.data
from sklearn.preprocessing import MultiLabelBinarizer
#from embedding import Embedding, RandomEmbedding, TextEmbedding, FastTextEmbedding
from utils import Index, AttributeHolder


from sklearn.datasets import fetch_20newsgroups
import pickle
    
'''

'''
class SemEval2010(torch.utils.data.Dataset):
  
  def __init__(self, path, subset = 'train.txt', nlines=None, maxseqlen=None, maxentlen=None, maxdist=60, nbos = 0, neos = 1, index = None, posiindex = None, classindex = None, rclassindex = None, dclassindex = None, eclassindex = None, compact=True):
    self.path = path
    self.subset = subset
    self.maxseqlen = maxseqlen
    self.maxdist = maxdist
    self.nbos = max(0, nbos)
    self.neos = max(1, neos)
    self.index = index if index is not None else Index()
    self.bosidx = self.index.add('<s>')
    self.eosidx = self.index.add('</s>')
    self.padidx = self.index.add('<pad>')
    self.epadidx = self.index.add('<epad>')
    self.classindex = classindex if classindex is not None else Index()
    self.rclassindex = rclassindex if rclassindex is not None else Index()
    self.dclassindex = dclassindex if dclassindex is not None else Index()
    self.eclassindex = eclassindex if eclassindex is not None else Index()
    self.posiindex = posiindex if eclassindex is not None else Index(initwords = [ maxdist, -maxdist ], unkindex = 0)
    self.maxentlen = maxentlen
    self.load(nlines, compact)
    self.device = torch.device('cpu')
    self.deviceTensor = torch.LongTensor().to(self.device) # create tensor on device, which can be used for copying
    
  @staticmethod
  def process_label(label):
    rval = AttributeHolder(label = label) # Product-Producer(e2,e1)
    rval.rlabel = re.sub(r'\(.*$', '', label) # Product-Producer
    rval.etypes = rval.rlabel.split('-')
    if len(rval.etypes) < 2:
      rval.dlabel = 'e1 - e2'
      rval.e1label = rval.rlabel
      rval.e2label = rval.rlabel
    else:
      rval.dlabel = re.sub(r'^[^(]+', '', label) # (e2,e1)
      idx_e1 = rval.dlabel.find('e1')
      idx_e2 = rval.dlabel.find('e2')
      e1_first = idx_e1 < idx_e2
      rval.e1label = rval.etypes[not e1_first] # == 0
      rval.e2label = rval.etypes[e1_first] # == 1
      rval.dlabel = 'e1 > e2' if e1_first else 'e1 < e2'
    return rval
  
  def label_to_index(self, labelobj):
    return AttributeHolder(
        label=self.classindex.add(labelobj.label),
        rlabel=self.rclassindex.add(labelobj.rlabel),
        dlabel=self.dclassindex.add(labelobj.dlabel),
        e1label=self.eclassindex.add(labelobj.e1label),
        e2label=self.eclassindex.add(labelobj.e2label))
    
  def get_offsets_spacy(self, row, col):
    doc = row.spacy
    # offsets e1, e2
    b = next((t.i for t in doc if t.idx >= row[col][0]), None)
    e = next((t.i for t in doc if t.idx >= row[col][1]), None)
    return (b, e)
  
  def transform_token(self, t):
    if isinstance(t, str):
      return t
    if t.pos_ == 'PUNCT' or t.pos_ == 'SYM' or len(t.lemma_.strip()) < 1:
      return None
    if t.pos_ == 'NUM':
      return '0'
    return t.text
#    if t.pos_ == 'NUM':
#      return '00#$'
#    return t.lemma_.lower() + '#' + t.tag_[0].upper()

  def make_sequence_tensor(self, row, placeholder_e = False):
    doc = row.spacy      

    # keep track of offsets e1 and e2
    temp_seq_e1 = [False] * len(doc)
    temp_seq_e1[row.offset_e1_spacy[0]:row.offset_e1_spacy[1]] = [True] * (row.offset_e1_spacy[1]-row.offset_e1_spacy[0])
    
    temp_seq_e2 = [False] * len(doc)
    temp_seq_e2[row.offset_e2_spacy[0]:row.offset_e2_spacy[1]] = [True] * (row.offset_e2_spacy[1]-row.offset_e2_spacy[0])
    
    s = map(self.transform_token, doc)
    s = zip(s, temp_seq_e1, temp_seq_e2)
    
    s = filter(lambda t : t[0] is not None, s)
    s = map(lambda t: (self.index.add(t[0]), t[1], t[2]), s)
    s = list(s)

    # add sentence begin and sentence end markers
    for i in range(self.nbos):
      s.insert(0, (self.bosidx, False, False))
    for i in range(max(self.neos, 1)):
      s.append((self.eosidx, False, False))
      
    seq, temp_seq_e1,  temp_seq_e2 = list(zip(*s))
    seq = torch.LongTensor(seq)
    row['e1_in_seq'] = temp_seq_e1.index(True)
    row['e1_in_seq'] = (row.e1_in_seq, temp_seq_e1.index(False, row.e1_in_seq))
    row['e2_in_seq'] = temp_seq_e2.index(True)
    row['e2_in_seq'] = (row.e2_in_seq, temp_seq_e2.index(False, row.e2_in_seq))
    row['seq_e1'] = seq[range(row.e1_in_seq[1]-1, row.e1_in_seq[0]-1, -1)] # save entities in reversed order
    row['seq_e2'] = seq[range(row.e2_in_seq[1]-1, row.e2_in_seq[0]-1, -1)]
    row['seqlen_e1'] = row.seq_e1.size(0)
    row['seqlen_e2'] = row.seq_e2.size(0)    

    if placeholder_e:
      e1 = self.index.add('<e1>')
      e2 = self.index.add('<e2>')
      seq = torch.cat((
          seq[:row.e1_in_seq[0]],
          torch.LongTensor([e1]),
          seq[row.e1_in_seq[1]:row.e2_in_seq[0]],
          torch.LongTensor([e2]),
          seq[row.e2_in_seq[1]:]))
      row['e1_in_seq'] = (seq == e1).nonzero()[0,0].item()
      row['e1_in_seq'] = (row.e1_in_seq, row.e1_in_seq+1)
      row['e2_in_seq'] = (seq == e2).nonzero()[0,0].item()
      row['e2_in_seq'] = (row.e2_in_seq, row.e2_in_seq+1)  
    row['seq'] = torch.LongTensor(seq)
    row['seqlen'] = seq.size(0)
    
    return row
  
  def pad(self, x, length, padval):
    y = torch.ones((length,)).long() * padval
    y[:min(len(x), length)] = x[:min(len(x), length)]
    return y
  
  def load(self, nlines, compact=True):
    source_file = os.path.join(self.path, self.subset)
    processed_file = source_file + '.pkl'
    if nlines:
      processed_file = processed_file + f'_{nlines:d}'      
    
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
          names=['id', 'originalsentence', 'labels', 'comment'],
          skip_blank_lines=True,
          encoding='utf-8',
          nrows=nlines)
      samples['original_offset_e1'] = samples.originalsentence.apply(lambda s: (s.find('<e1>'), s.find('</e1>')))
      samples['original_offset_e2'] = samples.originalsentence.apply(lambda s: (s.find('<e2>'), s.find('</e2>')))
      samples['e1'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e1[0] + 4:row.original_offset_e1[1]].strip(), axis=1)
      samples['e2'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e2[0] + 4:row.original_offset_e2[1]].strip(), axis=1)
      samples['l'] = samples.apply(lambda row: row.originalsentence[:row.original_offset_e1[0]].strip(), axis=1)
      samples['r'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e2[1]+5:].strip(), axis=1)
      samples['m'] = samples.apply(lambda row: row.originalsentence[row.original_offset_e1[1]+5:row.original_offset_e2[0]].strip(), axis=1)
      samples['sentence'] = samples.apply(lambda row: row.l+' '+row.e1+' '+row.m+' '+row.e2+' '+row.r, axis=1)
      samples['offset_e1'] = samples.apply(lambda row: (len(row.l)+1, len(row.l)+1+len(row.e1)), axis=1)
      samples['offset_e2'] = samples.apply(lambda row: (len(row.l)+1+len(row.e1)+1+len(row.m)+1, len(row.l)+1+len(row.e1)+1+len(row.m)+1+len(row.e2)), axis=1)
      samples['spacy'] = samples.sentence.progress_apply(lambda s: nlp(s))
      samples['offset_e1_spacy']= samples.apply(lambda r: self.get_offsets_spacy(r, 'offset_e1'), axis=1)
      samples['offset_e2_spacy']= samples.apply(lambda r: self.get_offsets_spacy(r, 'offset_e2'), axis=1)
      samples.to_pickle(processed_file)
      del samples

    # load processed messages
    self.samples = pandas.read_pickle(processed_file)    
    self.samples = self.samples.apply(lambda r: self.make_sequence_tensor(r, placeholder_e = True), axis=1)

    # pad
    if not self.maxseqlen:
      self.maxseqlen = self.samples.seqlen.max()      
    if not self.maxentlen:
      self.maxentlen = max(self.samples.seqlen_e1.max(), self.samples.seqlen_e2.max())
    
    self.samples['seq'] = self.samples.seq.apply(lambda s: self.pad(s, self.maxseqlen, self.padidx))
    self.samples['seqlen'] = self.samples.seqlen.apply(lambda l: min(l, self.maxseqlen))
    self.samples['e1_in_seq'] = self.samples.e1_in_seq.apply(lambda tup: (min(tup[0], self.maxseqlen), min(tup[1], self.maxseqlen)))
    self.samples['e2_in_seq'] = self.samples.e2_in_seq.apply(lambda tup: (min(tup[0], self.maxseqlen), min(tup[1], self.maxseqlen)))
    self.samples['seq_e1'] = self.samples.seq_e1.apply(lambda s: self.pad(s, self.maxentlen, self.epadidx))
    self.samples['seq_e2'] = self.samples.seq_e2.apply(lambda s: self.pad(s, self.maxentlen, self.epadidx))
    # create the position vectors (relative distance to the beginning of the entities)
    self.samples['e1_posi_seq'] = self.samples.e1_in_seq.apply(lambda offs_e1: torch.LongTensor(list(map(self.posiindex.add, map(lambda d: max(d, -self.maxdist) if d < 0 else min(d, self.maxdist), map(lambda i: i-offs_e1[0], range(self.maxseqlen)))))))
    self.samples['e2_posi_seq'] = self.samples.e2_in_seq.apply(lambda offs_e2: torch.LongTensor(list(map(self.posiindex.add, map(lambda d: max(d, -self.maxdist) if d < 0 else min(d, self.maxdist), map(lambda i: i-offs_e2[0], range(self.maxseqlen)))))))

    # prep labels
    self.samples['labels'] = self.samples.labels.apply(self.process_label)
    self.samples['labelids'] = self.samples.labels.apply(self.label_to_index)
    
    # reconstructed sequence for debugging purposes
    self.samples['seq_recon'] = self.samples.seq.apply(lambda t: np.array(list(self.index[t.tolist()])))
          
    return True
  
  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, index):
    r  = self.samples.iloc[index]
    s  = r.seq
    sl = r.seqlen

    e1_posi_seq = r.e1_posi_seq
    e2_posi_seq = r.e2_posi_seq    
    
    e1_offs = r.e1_in_seq
    e1      = r.seq_e1
    e1_len  = r.seqlen_e1
    e2_offs = r.e2_in_seq
    e2      = r.seq_e2
    e2_len  = r.seqlen_e2
    
    # left, right and middle as indices (from,to)
    left  = torch.LongTensor([0, e1_offs[0]])
    right = torch.LongTensor([e2_offs[1], sl])
    mid   = torch.LongTensor([e1_offs[1], e2_offs[0]])
    
    label   = r.labelids.label
    e1label = r.labelids.e1label
    e2label = r.labelids.e2label
    rlabel  = r.labelids.rlabel
    dlabel  = r.labelids.dlabel
        
    d = self.deviceTensor    
    return d.new_tensor(index), d.new_tensor(r.id), d.new_tensor(s), d.new_tensor(sl), d.new_tensor(e1_posi_seq), d.new_tensor(e2_posi_seq),  d.new_tensor(left), d.new_tensor(e1_offs), d.new_tensor(mid), d.new_tensor(e2_offs), d.new_tensor(right), d.new_tensor(e1), d.new_tensor(e1_len), d.new_tensor(e2), d.new_tensor(e2_len), d.new_tensor(label), d.new_tensor(e1label), d.new_tensor(e2label), d.new_tensor(rlabel), d.new_tensor(dlabel)
  
  def cpu(self):
    return self.to(torch.device('cpu'))
  
  def cuda(self):
    return self.to(torch.device('cuda'))
  
  def to(self, device):
    print(f"Impossible to switch device! Whatever you're sayin, I stay on CPU! But, for you, I will send new Tensors to `{device}`. Sincerely yours, {self.__class__.__name__:s}.", file=sys.stderr)
    self.device = device
    self.deviceTensor = self.deviceTensor.to(device)
    return self
  
  def __repr__(self):
    return f'''\
{self.__class__.__name__:s} (
  path: {self.path:s}
  subset: {self.subset:s}
  nsamples: {self.samples.shape[0]:d}
  maxseqlen: {self.maxseqlen:d}
  maxentlen: {self.maxentlen:d}
  maxdist: {self.maxdist:d}
  wordindex: {self.index}
  posiindex: {self.posiindex}
  classindex: {self.classindex}
  rclassindex: {self.rclassindex}
  dclassindex: {self.dclassindex}
  eclassindex: {self.eclassindex}
  device: {self.device}
  sample[0]: {self.samples.iloc[0].seq_recon}
)  
'''
  
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


