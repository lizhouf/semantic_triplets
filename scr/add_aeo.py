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

'''
Example Key Words
'''
evaluation_verbs = ["feel","smell","taste","look","hear","see","think","know"]
orientation_verbs = ["remember","bear","grow","belong"]
imagine_verbs = ["want","should","would","could","can","might","may"]

'''
AEO
'''

def add_aeo_df(df): # chunk_tri df
    cat_list = []

    for i in range(len(df)):

        # initiate the rule components
        this_rel = df.relations[i]
        this_obj = df.objects[i]

        rel_has_evaluation = 0
        rel_has_orientation = 0
        rel_has_imagine = 0

        rel_has_be = 0
        rel_has_have = 0
        rel_has_to = 0

        rel_has_neg = 0

        rel_has_VBG = 0
        rel_num_verb = 0

        obj_is_adj = 0  # only adj, no NOUN+

        obj_has_no = 0

        # give value
        for rel in this_rel:

            # rel lemmas
            try:
                if rel.lemma_ in evaluation_verbs:
                    rel_has_evaluation = 1
                elif rel.lemma_ in imagine_verbs:
                    rel_has_imagine = 1
                elif rel.lemma_ in orientation_verbs:
                    rel_has_orientation = 1
                elif rel.lemma_ == "be":
                    rel_has_be = 1
                elif rel.lemma_ == "have":
                    rel_has_have = 1
                elif rel.lemma_ == "to":
                    rel_has_to = 1
            except:  # avoid no lemma
                0

            # rel dep
            try:
                if rel.dep == neg:
                    rel_has_neg = 1
            except:
                0

            # rel pos
            try:
                if rel.pos == VERB:
                    rel_num_verb = rel_num_verb + 1
            except:
                0

            # rel tag
            try:
                if rel.tag_ == "VBG":
                    rel_has_VBG = 1
            except:
                0

        for obj in this_obj:
            # obj lemma
            try:
                if obj.lemma_ == "no":
                    obj_has_no = 1
            except:
                0

        for obj in this_obj:  # seperate, want to break
            # obj pos
            try:
                if obj.pos == ADJ:
                    obj_is_adj = 1
                if obj.pos in [NOUN,PRON,PROPN]:
                    obj_is_adj = 0
                    break
            except:
                0

        # judge:

        # fixed words
        if rel_has_evaluation and obj_is_adj:
            cat_list.append("Evaluation")
            continue
        if rel_has_imagine:
            cat_list.append("Agency_Possible")
            continue
        if rel_has_orientation:
            cat_list.append("Orientation")
            continue

        # neg
        if rel_has_neg or obj_has_no:
            cat_list.append("Orientation")
            continue

        # have
        if rel_has_have:
            if rel_has_to:
                cat_list.append("Agency_Coercive")
                continue
            else:
                cat_list.append("Orientation")
                continue

        # be
        if rel_has_be:
            if obj_is_adj:
                cat_list.append("Evaluation")
                continue
            elif rel_has_VBG:
                cat_list.append("Agency_Active")
                continue
            elif rel_num_verb > 1:
                cat_list.append("Agency_Passive")
                continue
            elif rel_num_verb == 1:
                cat_list.append("Orientation")
                continue

        # if none of the above, then judge:
        cat_list.append("Agency_Active")

    df["AEO_cat"] = cat_list

    return df
