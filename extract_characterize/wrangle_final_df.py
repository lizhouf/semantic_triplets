'''
import packages
'''
import os
# data manipulation
import pandas as pd
import numpy as np
# Text manipulation
import re
# Load your usual SpaCy model (one of SpaCy English models)
import spacy
nlp = spacy.load('en_core_web_lg')
nlps = spacy.load('en_core_web_lg')
from spacy.matcher import Matcher
# more on Spacy
from spacy.symbols import dobj, obj, pobj, acomp, ccomp, pcomp, xcomp, conj, acomp, ccomp, pcomp, xcomp, advmod, amod
from spacy.symbols import neg, det, aux, prep, poss, nsubj, nsubjpass, csubj, csubjpass, det, prt
from spacy.symbols import VERB, DET, ADP, ADV, ADJ, NOUN, PRON, PROPN, PART
# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)
# spacy +
import textacy
# for copy a list in df
import copy
# visual
import networkx as nx
import ast
import matplotlib.pyplot as plt
# cluster
from sklearn import cluster, datasets
import json
import sys
import codecs
from nltk.corpus import wordnet
from collections import defaultdict, Counter
import collections
# cluster 2
from nltk.corpus import wordnet as wn
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
# import matplotlib.pyplot as plt
# import matplotlib as mpl
from gensim.models import KeyedVectors
# cosine similarity
from scipy import spatial

'''
Speical Lists
'''
wh_list = ["who","what","how","why","where","when","whom","Who","What","How","Why","Where","When","Whom"]

'''
Helpers for post-moderation
'''

# helper: turn subjects, relations, and objects into string
def to_string(the_list):
    the_string = ""
    for token in the_list:
        the_string = the_string + token.text + " "
    return the_string[:-1]  # get rid of the last

# helper: clean non-English characters
# remove anything other than . or - or ?
# use in subject, relation, and object cleaning
def remove_non_english(the_str):
    the_str = the_str.replace("n't"," not")
    the_str = the_str.replace("'s"," is")
    the_str = re.sub(r"[^\s0-9a-zA-Z.?-]", ' ', the_str)
    the_str = re.sub(r"\s+", ' ', the_str)
    return the_str

# helper: clean repetitive words
# use in subject, relation, and object cleaning
# # reference: https://www.geeksforgeeks.org/remove-consecutive-duplicates-string/
def remove_repetitive(S):
    # split and get length
    S=S.split()
    n = len(S)
    # We don't need to do anything for empty or single character string.
    if (n < 2):
        return " ".join(S)
    # j is used to store index is result string (or index of current distinct character)
    j = 0
    # Traversing string
    for i in range(n):
        # If current character S[i] is different from S[j]
        if (S[j] != S[i]):
            j += 1
            S[j] = S[i]
    # Putting string termination character.
    j += 1
    S = S[:j]
    # append back and return
    return " ".join(S)

'''
main
'''
def final_wrangle_df(df):
    # remove repetitive and non-English
    df.subjects = df.subjects.apply(lambda x: remove_repetitive(remove_non_english(to_string(x))))
    df.relations = df.relations.apply(lambda x: remove_repetitive(remove_non_english(to_string(x))))
    df.objects = df.objects.apply(lambda x: remove_repetitive(remove_non_english(to_string(x))))

    # remove wh subjects
    df["sub_has_wh"] = df.subjects.apply(lambda x: 1 if any(ele in x for ele in wh_list) else 0)
    df = df[df.sub_has_wh == 0].reset_index(drop=True)

    # remove empty
    df = df[df.objects != ""].reset_index(drop=True)

    return df