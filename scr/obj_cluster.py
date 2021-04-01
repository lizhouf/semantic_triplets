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
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
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
# tokenizer
# reference: https://stackoverflow.com/questions/58105967/spacy-tokenization-of-hyphenated-words
def custom_tokenizer(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                            suffix_search=nlp.tokenizer.suffix_search,
                            infix_finditer=infix_re.finditer,
                            token_match=nlp.tokenizer.token_match,
                            rules=nlp.Defaults.tokenizer_exceptions)
nlp.tokenizer = custom_tokenizer(nlp)
# import matplotlib.pyplot as plt
# import matplotlib as mpl
from gensim.models import KeyedVectors
# cosine similarity
from scipy import spatial


dict_path = "/Users/lizhoufan/Dropbox/HGSDLab/Ch6/Data/Raw/" # need to change directory later
cat_dict = pd.read_excel(dict_path+"Shoah_cat_list_tp.xlsx") # v2 is more general
cat_dict.keywords = cat_dict.keywords.apply(lambda x: str(x).strip().replace(",","").split(" ")) # change to lists

def find_wn_noun(word):
    for i in range(len(wn.synsets(word))):
        name = wn.synsets(word)[i].name()
        if name.split(".")[1]=="n":
            return wn.synsets(word)[i]


def find_shoah_cat(this_word,cat_dict=cat_dict):
    if 1:
        if this_word.pos_ == "NOUN":
            try:
                wordnet_name = this_word._.wordnet.synsets()[0].name()
            except:
                return None
            category_list=[]
            similarity_list=[]

            if wordnet_name.split(".")[1]=="n":
                for i in range(len(cat_dict)):
                    for j in range(len(cat_dict.keywords_synset[i])):
                        shoah_word_syn = cat_dict.keywords_synset[i][j]
                        #print(shoah_word_syn)
                        this_word_syn = this_word._.wordnet.synsets()[0]
                        this_similarity = this_word_syn.path_similarity(shoah_word_syn)
                        
                        category_list.append(cat_dict.shoah_cat[i])
                        similarity_list.append(this_similarity)

            return category_list[similarity_list.index(max(similarity_list))]

cat_dict["keywords_synset"] = cat_dict.keywords.apply(lambda x: [find_wn_noun(y) for y in x])
cat_dict.keywords_synset = cat_dict.keywords_synset.apply(lambda x: [i for i in x if i])

def add_obj_based_cluster(df): # df MUST have column "objects_tokens"

    shoah_cats = []
    for i in range(len(df)):
        category = ""
        for j in range(len(df.objects_tokens[i])): # keep the category of the first element if 2+ nouns
            word = df.objects_tokens[i][j]
            #print(word)
            category = ""
            try:
                category = find_shoah_cat(word,cat_dict=cat_dict)
                #print("Success.")
                continue # the first -> break # if the last -> continue
            except:
                #print("NOT Success.")
                0
        
        if category=="Others":
            category = ""

        shoah_cats.append(category)
            
    df["shoah_cats"] = shoah_cats

    return df

