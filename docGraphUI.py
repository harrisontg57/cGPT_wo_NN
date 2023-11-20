import Graph_Learner
from Graph_Learner import get_word_seq, seq_to_sentence, Bloom_Filter, doc_graph
import sys
import random

random.seed("Test42Te")

if len(sys.argv) > 1:
    #Take input
    training_file = sys.argv[1]
    f = open(training_file, 'r', encoding="utf-8")
    text = f.read()
    doc = get_word_seq(text)
    if len(sys.argv) > 2:
        h = int(sys.argv[2][2:])
    if len(sys.argv) > 3:
        mode = sys.argv[3][1:]
else:
    #Request Input
    training_file = input("Add Training File:")
    f = open(training_file, 'r', encoding="utf-8")
    text = f.read()
    doc = get_word_seq(text)
    #Add support for inputting h and mode

#Train the graph
g = doc_graph(h,mode)
g.add_doc(doc)

prompt_in = input("Add Prompt (Either file or a text sequence in quotes..):")
if prompt_in[0] == '"':
    prompt = get_word_seq(prompt_in)
else:
    fp = open(prompt_in, 'r', encoding="utf-8")
    pt = fp.read()
    prompt = get_word_seq(pt)

n = int(input("How many new words to generate:"))

out = g.gen_next_n(prompt,n,h)
print(seq_to_sentence(out))

while True:
    new_in= input("Add Word(s) in Quotes followed by N to generate N new words (if N > 1):")
    s_in = new_in.split('"')
    nprompt = get_word_seq(s_in[1])
    out.extend(nprompt)
    if len(s_in) == 3:
        nn = int(s_in[-1])
    else:
        nn = 1
    out = g.gen_next_n(out,nn,h)
    print(seq_to_sentence(out))
