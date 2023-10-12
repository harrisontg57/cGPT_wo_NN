import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import random

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_word_seq(text):
  seq = []
  for t in word_tokenize(text):
    if t.isalnum() or t == ".":
      seq.append(t.lower())
  return seq
  
def seq_to_sentence(seq):
  return ' '.join(seq)

class Bloom_Filter:
  def __init__(self, m=5000):
    self.bit_vector = [0]*m
    self.m = m

  def k1(self,seq):
    k1 = hash(seq[-1])%self.m
    if k1 < 0:
      return self.m + k1
    else:
      return k1
  
  def k_func(self,seq,selected):
    #selected is an array of which elements in seq to use in hashing, counting
    #backwards from the last element (The most recently visted node)
    toHash = []
    lseq = len(seq)
    for s in selected:
      if s > lseq: break
      toHash.append(seq[-s])
    k_f = hash(tuple(toHash))%self.m
    if k_f < 0:
      return self.m + k_f
    else:
      return k_f

  def k_all(self,seq):
    k1 = hash(seq[-1])%self.m
    if k1 < 0:
      return self.m + k1
    else:
      return k1

  def addto(self,seq):
    if seq[0] == None:
      seq = seq[1:]
      #Remove None element at start if needed.
    lseq = len(seq)
    self.bit_vector[self.k1(seq)] = 1
    self.bit_vector[self.k_all(seq)] = 1
    self.bit_vector[self.k_func(seq,[2])] = 1
    self.bit_vector[self.k_func(seq,[2,3,4])] = 1

  def addtoall(self,seq):
    if seq[0] == None:
      seq = seq[1:]
    lseq = len(seq)
    for x in range(lseq-1,-1,-1):
      self.addto(seq[x:lseq])

  def check(self,seq):
    if seq[0] == None:
      seq = seq[1:]
    if self.bit_vector[self.k1(seq)] == 0: return False
    if self.bit_vector[self.k_func(seq,[2])] == 0: return False
    if self.bit_vector[self.k_func(seq,[2,3,4])] == 0: return False
    if self.bit_vector[self.k_all(seq)] == 0: return False
    return True

class doc_graph:
  def __init__(self,h=5,gen_mode='max'):
    self.h = h
    self.gen_mode = gen_mode
    self.edge_table = {}
    self.node_to_edge_table = {}
    self.edge_to_bfilter_table = {}
  
  def set_gen_mode(self,gen_mode):
    self.gen_mode = gen_mode

  def add_doc(self,doc,h=5):
    
    for i,w in enumerate(doc[:-1]):
      key = (w,doc[i+1])
      if w in self.node_to_edge_table:
        self.node_to_edge_table[w] = self.node_to_edge_table[w] + (key,)
      else:
        self.node_to_edge_table[w] = (key,)
      if key in self.edge_table:
        hashes = set(self.edge_table[key])
        bfilter = self.edge_to_bfilter_table[key]
      else:
        hashes = set()
        bfilter = Bloom_Filter()
      for k in range(0,h):
        if k >= i: break
        if i < h:
          hashes.add(hash((None,w)))
          hashable = (None,) + tuple(doc[k:i+1]) #Check that this is right...
          hashes.add(hash(hashable))
          bfilter.addto(hashable)
        else:
          hashes.add(hash((None,w)))
          hashable = (None,) + tuple(doc[i-k:i+1])
          hashes.add(hash(hashable))
          bfilter.addto(hashable)
      
      self.edge_to_bfilter_table[key] = bfilter
      self.edge_table[key] = hashes
    #return doc
  
  def add_text(self,prompt,text,h=5):
    if len(prompt) > h:
      prompt[-h:].extend(text)
    else:
      prompt.extend(text)
    self.add_doc(prompt,h)
    return prompt
  
  def add_text_and_gen_n(self,prompt,text,n,retrain=False,h=5):
    #Add text to the current sequence and generate n new elements
    #If retrain is True text and the last h elements of prompt are 
    #trained into the graph as if they were a new document.
    if retrain: prompt = self.add_text(prompt,text,h)
    else: prompt.extend(text)
    return self.gen_next_n(prompt,n,h)

  def gen_next_n(self,prompt,n,h=5):
    for x in range(n):
      prompt = self.gen_next(prompt,h)
    return prompt

  def gen_next(self,prompt,h=5):
    ltext = len(prompt)
    if ltext < h:
      seq = (None,) + tuple(prompt)
    elif ltext > h:
      seq = tuple(prompt[-h:])
    else:
      seq = tuple(prompt)
    assert seq[-1] in self.node_to_edge_table, "Last Item in Sequence Not Encountered in Training"
      #This needs to handle errors where a node was not encountered in training
      #and therefore has no associated edges..
    hseq = hash(seq)
    edges = self.check_seq(seq)
    if len(edges) == 0:
      for k in range(len(seq)):
        edges = self.check_seq(seq[k:])
        if len(edges) > 0: break
    if len(edges) == 1:
      prompt.append(edges[0][1])
      return prompt
    if len(edges) > 1:
      if self.gen_mode == 'max':
        #Select the edge in edges with the max Pi
        emax = 0
        edge = (None,None)
        for e in edges:
          x = len(self.edge_table[e])
          if x > emax: 
            emax = x
            edge = e
        prompt.append(edge[1])
      elif self.gen_mode == 'rand':
        #Randomly select from the set of edges.
        edge = random.choice(edges)
        prompt.append(edge[1])

      return prompt
    
  def check_seq(self,seq):
    node_key = seq[-1]
    edges = self.node_to_edge_table[node_key]
    fedges = []
    for e in edges:
      bfilter = self.edge_to_bfilter_table[e]
      if bfilter.check(seq):
        fedges.append(e)
    return fedges
