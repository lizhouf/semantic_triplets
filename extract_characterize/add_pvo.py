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
Annotated Lists
'''

# after the human annotations
v_final = pd.read_excel("/Users/lizhoufan/Dropbox/Holocaust_DS/Triplets/victim_final_annotated.xlsx")
p_final = pd.read_excel("/Users/lizhoufan/Dropbox/Holocaust_DS/Triplets/perpetrator_final_annotated.xlsx")

# 6 levels
v_3 = list(v_final[v_final.certainty == 3].lemma)
v_2 = list(v_final[v_final.certainty == 2].lemma)
v_1 = list(v_final[v_final.certainty == 1].lemma)
p_3 = list(p_final[p_final.certainty == 3].lemma)
p_2 = list(p_final[p_final.certainty == 2].lemma)
p_1 = list(p_final[p_final.certainty == 1].lemma)

'''
Helper functions for wrangling
'''

def sub_lemma_no_pron(token_list):
    the_string = ""
    for token in token_list:
        try:
            if token.lemma_ == "-PRON-":
                if str(token) == "I":
                    the_string = the_string + str(token) + " "
                else:
                    the_string = the_string + str(token).lower() + " "
            else:
                the_string = the_string + token.lemma_.lower() + " "
        except:
            the_string = the_string + str(token).lower() + " "
    return the_string[:-1]  # get rid of the last


def rel_lemma_no_pron(token_list):
    the_string = ""
    for token in token_list:
        try:
            the_string = the_string + token.lemma_.lower() + " "
        except:
            the_string = the_string + str(token) + " "
    return the_string[:-1]  # get rid of the last


def obj_lemma_no_pron(token_list):
    the_string = ""
    for token in token_list:
        try:
            if token.pos_ not in ["PRON","NOUN","PROPN","ADJ"]:
                continue
            elif token.lemma_ == "-PRON-":
                the_string = the_string + str(token) + " "
            else:
                the_string = the_string + token.lemma_ + " "
        except:
            the_string = the_string + str(token) + " "
    return the_string[:-1]  # get rid of the last

'''
PVO
'''

def add_pvo(the_string,v_3,v_2,v_1,p_3,p_2,p_1):  # perpetrator, victim, other

    PVO_label = "O"
    lemma_no_pron = the_string.split(" ")

    for i in range(len(lemma_no_pron)):
        # search 6 levels of dictionaries
        if any(x in lemma_no_pron for x in p_3):
            PVO_label = "P"
            continue
        elif any(x in lemma_no_pron for x in v_3):
            PVO_label = "V"
            continue
        elif any(x in lemma_no_pron for x in p_2):
            PVO_label = "P"
            continue
        elif any(x in lemma_no_pron for x in v_2):
            PVO_label = "V"
            continue
        elif any(x in lemma_no_pron for x in p_1):
            PVO_label = "P"
            continue
        elif any(x in lemma_no_pron for x in v_1):
            PVO_label = "V"
            continue

    return PVO_label

def add_pvo_df(df):
    # convert Spacy objects to Strings
    df["subjects_lemma"] = df.subjects.apply(lambda x: sub_lemma_no_pron(x))
    df["relations_lemma"] = df.relations.apply(lambda x: rel_lemma_no_pron(x))
    df["objects_lemma"] = df.objects.apply(lambda x: obj_lemma_no_pron(x))

    # add PVO_cat
    df["PVO_cat_sub"] = df.subjects_lemma.apply(
        lambda x: add_pvo(x,v_3=v_3,v_2=v_2,v_1=v_1,p_3=p_3,p_2=p_2,p_1=p_1))
    df["PVO_cat_obj"] = df.objects_lemma.apply(
        lambda x: add_pvo(x,v_3=v_3,v_2=v_2,v_1=v_1,p_3=p_3,p_2=p_2,p_1=p_1))

    # print("PVO",df.head())

    return df
