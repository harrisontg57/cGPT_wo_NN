import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import random
import pickle

try:
    #Check for punkt dataset and download if it doesn't not exist
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_word_seq(text):
    #convert text document into a list of words.
  seq = []
  for t in word_tokenize(text):
    if t.isalnum() or t == ".":
      seq.append(t.lower())
  return seq

def seq_to_sentence(seq):
    #convert a sequence of words into a sentence.
  return ' '.join(seq)

class Bloom_Filter_Mem_Eff:
    #Memory efficent implementation of bloomfilter. Stores hashes as keys in a dictionary.
  def __init__(self, m=15000):
    self.table = {}
    self.m = m

  def k1(self,seq):
    #k1 is a hash of just the final element of the sequence
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
    #k_all hashes all elements in the sequence
    k1 = hash(seq[-1])%self.m
    if k1 < 0:
      return self.m + k1
    else:
      return k1

  def addto(self,seq):
    #Add a sequence to the bloom filter.
    if seq[0] == None:
      seq = seq[1:]
      #Remove None element at start if needed.
    lseq = len(seq)
    self.table[self.k1(seq)] = 1
    self.table[self.k_all(seq)] = 1
    self.table[self.k_func(seq,[2])] = 1
    self.table[self.k_func(seq,[2,3,4])] = 1

  def addtoall(self,seq):
    #Add sequence and its subsequences to the bfilter.
    if seq[0] == None:
      seq = seq[1:]
    lseq = len(seq)
    for x in range(lseq-1,-1,-1):
      self.addto(seq[x:lseq])

  def check(self,seq):
    #Check if a sequence is in the bloom filter
    if seq[0] == None:
        #remove None element from start of sequence
      seq = seq[1:]
    if self.k1(seq) not in self.table: return False #First check if the final element hashes since this eliminates the need to perform the other k functions
    if self.k_func(seq,[2]) not in self.table: return False
    if self.k_func(seq,[2,3,4]) not in self.table: return False
    if self.k_all(seq) not in self.table: return False
    return True

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
    #Add sequence and its subsequences to the bfilter.
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
    #Initiate graph, assigning generate mode
    self.h = h #Currently serves no purpose but can be adapted to serve as an h_max, the maximum h allowed during generation
    self.gen_mode = gen_mode
    #Create tables of edges and nodes
    self.edge_table = {}
    #edge_table stores edges and their associated Pi the set of hashed sequences encountered during training
    self.node_to_edge_table = {}
    #Node_to_edge_table stores nodes and their associated edges
    self.edge_to_bfilter_table = {}
    #Edge_to_bfilter_table stores edges and their associated bloom filters.

  def set_gen_mode(self,gen_mode):
    #Sets the text generation mode
    self.gen_mode = gen_mode

  def add_doc(self,doc,h=5):
    #Add a document and train a graph (if this is the first document added)
    #If graph already exists it trains in new data from doc
    ldoc = len(doc)
    for i,w in enumerate(doc[:-1]):
      if i%12500 == 0:
        print(str(100.0*(i+1)/ldoc)+'% Complete.')
        print(str(i) + ' of ' + str(ldoc))
      #Iterate through the document looking at each word w
      key = (w,doc[i+1])
      #Create a key for edge (W(n),W(n+1))
      if w in self.node_to_edge_table:
        #if w has already been seen, load its associated edge set
        self.node_to_edge_table[w] = self.node_to_edge_table[w] + (key,)
      else:
        #otherwise create a new edge set
        self.node_to_edge_table[w] = (key,)
      if key in self.edge_table:
        #if edge has been seen before load its hash set Pi and bloom filter
        hashes = set(self.edge_table[key])
        bfilter = self.edge_to_bfilter_table[key]
      else:
        #otherwise create these.
        hashes = set()
        bfilter = Bloom_Filter_Mem_Eff2()

      hashes.add(hash((None,w)))
      bfilter.addto((None,w))
      #print(key)
      for k in range(h-1,0,-1):
        #For k in depth h build sequences and subsequences of (W(n-h),...,W(n))
        if k >= i: continue #At start of doc k might exceed the word index, so stop building sequences
        if i < h:
            #if index is less than h at None type to start of sequence.
            #Then generate Pi

          hashable = (None,) + tuple(doc[k:i+1])
          hashes.add(hash(hashable))
          bfilter.addto(hashable)
        else:
            #Generate the hash of the sequence (W(n-k),...,W(n))
            #Add it to Pi and to the bloom filter.
          if k < h-1:
            hashable = (None,) + tuple(doc[i-k:i+1])
          else:
            hashable = tuple(doc[i-k:i+1])

          #print(hashable)
          hashed = hash(hashable)
          if hashed in hashes: break  #Stop adding sequences since we've seen this pattern at this edge before.
          hashes.add(hashed)
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
    #Given a prompt and n return that prompt + n new words
    #Iteratively calls gen_next() n times.
    for x in range(n):
      prompt = self.gen_next(prompt,h)
    return prompt

  def gen_next(self,prompt,h=5):
    #Given a prompt generate the next word and return prompt + that word.
    ltext = len(prompt)
    if ltext < h:
        #Adds none type ot start of seq when prompt is < h
      seq = (None,) + tuple(prompt)
    elif ltext > h:
        #Removes words further back than h
      seq = tuple(prompt[-h:])
    else:
      seq = tuple(prompt)
    assert seq[-1] in self.node_to_edge_table, "Last Item in Sequence Not Encountered in Training"
      #Throws error when a node was not encountered in training
      #and therefore has no associated edges.
    hseq = hash(seq)
    edges = self.check_seq(seq)
    #get edges which have positive bloom filter responses for seq
    if len(edges) == 0:
        #If no edges had a positive filter response try sequence without its first item.
        #Repeat until an edge(s) has a positive response.
      for k in range(len(seq)):
        edges = self.check_seq(seq[k:])
        if len(edges) > 0: break
    if len(edges) == 1:
        #If only 1 edge has a positive filter response its associated word is predicted.
      prompt.append(edges[0][1])
      return prompt
    if len(edges) > 1:
        #If more than 1 edge has a positive response choose based on generation mode.
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
    #given a sequence load the bloom filters associated with its last item
    #then check which may contain the sequence
    #return fedges the list of edges which returned a positive from their bloom filter.
    node_key = seq[-1]
    edges = self.node_to_edge_table[node_key]
    fedges = []
    for e in edges:
      bfilter = self.edge_to_bfilter_table[e]
      if bfilter.check(seq):
        fedges.append(e)
    return fedges

def pickle_graph(g,filename='default'):
    #Save graph g as a python pickle type
  fend = '.pickle'
  name = filename + fend
  toPickle = [g.edge_table,g.node_to_edge_table,g.edge_to_bfilter_table]
  with open(name, 'wb') as handle:
    pickle.dump(toPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_to_graph(g,filename):
    #load graph values from pickle into graph g
  fend = '.pickle'
  name = filename + fend
  with open(name, 'rb') as handle:
    unpickled = pickle.load(handle)
    g.edge_table = unpickled[0]
    g.node_to_edge_table = unpickled[1]
    g.edge_to_bfilter_table = unpickled[2]

class Bloom_Filter_Mem_Eff2:
    #Memory efficent implementation of bloomfilter. Stores hashes as keys in a dictionary.
  def __init__(self, m=150000):
    self.table = set()
    self.m = m

  def k1(self,seq):
    #k1 is a hash of just the final element of the sequence
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
    #k_all hashes all elements in the sequence
    k1 = hash(seq)%self.m
    if k1 < 0:
      return self.m + k1
    else:
      return k1

  def addto(self,seq):
    #Add a sequence to the bloom filter.
    if seq[0] == None:
      seq = seq[1:]
      #Remove None element at start if needed.
    lseq = len(seq)
    #self.table.add(self.k1(seq))
    self.table.add(self.k_all(seq))
    self.table.add(self.k_func(seq,[2]))
    self.table.add(self.k_func(seq,[2,3,4]))

  def addtoall(self,seq):
    #Add sequence and its subsequences to the bfilter.
    if seq[0] == None:
      seq = seq[1:]
    lseq = len(seq)
    for x in range(lseq-1,-1,-1):
      self.addto(seq[x:lseq])

  def check(self,seq):
    #Check if a sequence is in the bloom filter
    if seq[0] == None:
        #remove None element from start of sequence
      seq = seq[1:]
    #if self.k1(seq) not in self.table: return False #First check if the final element hashes since this eliminates the need to perform the other k functions
    if self.k_func(seq,[2]) not in self.table: return False
    if self.k_func(seq,[2,3,4]) not in self.table: return False
    if self.k_all(seq) not in self.table: return False
    return True

def merge_bfilter(a,b):
  bfilter = Bloom_Filter_Mem_Eff2()
  bfilter.table = a.table | b.table
  return bfilter

def merge_graphs(a,b):
  g = doc_graph(a.h,a.gen_mode)
  edges = a.edge_table.keys() | b.edge_table.keys()
  nodes = a.node_to_edge_table.keys() | b.node_to_edge_table.keys()
  node_to_edge_table = {}
  edge_table = {}
  for node in nodes:
    if node in a.node_to_edge_table and node in b.node_to_edge_table: #ADD CHECK TO SEE IF NODE IS IN BOTH DICTS  DO AGAIN FOR EDGES
      g.node_to_edge_table[node] = set(a.node_to_edge_table[node]) | set(b.node_to_edge_table[node])
    elif node in a.node_to_edge_table:
      g.node_to_edge_table[node] = a.node_to_edge_table[node]
    else:
      g.node_to_edge_table[node] = b.node_to_edge_table[node]
  for edge in edges:
    if edge in a.edge_table.keys() and edge in b.edge_table.keys():
      g.edge_table[edge] = set(a.edge_table[edge]) | set(b.edge_table[edge])
      g.edge_to_bfilter_table[edge] = merge_bfilter(a.edge_to_bfilter_table[edge],b.edge_to_bfilter_table[edge])
    elif edge in a.edge_table.keys():
      g.edge_table[edge] = a.edge_table[edge]
      g.edge_to_bfilter_table[edge] = a.edge_to_bfilter_table[edge]
    else:
      g.edge_table[edge] = b.edge_table[edge]
      g.edge_to_bfilter_table[edge] = b.edge_to_bfilter_table[edge]
  return g

def clean_seq(original_seq):
  seq = []
  for t in original_seq:
    if t.isalnum() or t == ".":
      seq.append(t.lower())
  return seq
