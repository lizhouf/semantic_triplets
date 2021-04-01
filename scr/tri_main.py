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

# import others
from tri_chunk_extract import *
from add_aeo import *
from obj_cluster import *

'''
Preprocessing
'''

def clean_text(text):
    '''
    Clean the text.
    Args:
        text -> Str
    Returns:
        A cleaned string.
    '''
    # periods
    cleaned = text.replace('...',' ... ')
    cleaned = cleaned.replace('. . .',' ... ')
    # slashes
    cleaned = cleaned.replace('--',' -- ')
    # brackets
    brackets_regex = re.compile(r'\([^)]*\)')
    cleaned = re.sub(brackets_regex,' ',cleaned)
    # language tag (not for boder)
    language_tag_pattern = re.compile(r'\[.*?\]')
    cleaned = re.sub(language_tag_pattern,' ',cleaned)
    # space (tabs, newlines, etc)
    space_regex = re.compile(r'\s\s+')
    cleaned = re.sub(space_regex,' ',cleaned)

    return cleaned

def text2df(text):
    '''
    Convert text to df.
    Args:
        text -> Str
    Returns:
        A cleaned string.
    '''
    
    # get rid of - and things in between
    # get rid of things in between ""
    sentence_pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    interviewee_list = re.split(sentence_pattern,str(text))

    # convert to df
    df = pd.DataFrame({"Texts": interviewee_list})
    df = df[df.Texts != ""].reset_index(drop=True)

    return df

'''
Post-moderation
'''

# special list
wh_list = ["who","what","how","why","where","when","whom","Who","What","How","Why","Where","When","Whom"]

def to_string(the_list):
    '''
    Turn subjects, relations, and objects into string.
    Args:
        spaCy Doc -> Str
    Returns:
        A string.
    '''
    the_string = ""
    for token in the_list:
        the_string = the_string + token.text + " "
    return the_string[:-1]  # get rid of the last

# helper: clean non-English characters
# remove anything other than . or - or ?
# use in subject, relation, and object cleaning

def remove_non_english(the_str,change_is):
    '''
    Remove non-English characters and 
    Clean abb. expressions.
    Args:
        the_str -> string
        change_is -> bool
    Returns:
        A string.
    '''

    the_str = the_str.replace("ca n’t","can not") # no meta boder version
    the_str = the_str.replace("ca n't","can not")

    the_str = the_str.replace("n't"," not") # no meta boder version
    the_str = the_str.replace("n’t"," not")

    the_str = the_str.replace("'m"," am")
    the_str = the_str.replace("I m","I am")
    the_str = the_str.replace("'re"," are")

    if change_is==True:
        the_str = the_str.replace("'s"," is")
        if the_str == "s":
            the_str = "is"

    the_str = the_str.replace("'d"," would")
    the_str = the_str.replace("'ll"," will")
    the_str = the_str.replace("o'clock","of the clock")

    the_str = re.sub(r"[^\s0-9a-zA-Z.?-]", ' ', the_str)
    the_str = re.sub(r"\s+", ' ', the_str)

    return the_str 

def remove_repetitive(S):
    '''
    Clean repetitive words in use in subjects, relations, and objects.
    Args:
        S -> string
    Returns:
        A string.
    Reference:
        https://www.geeksforgeeks.org/remove-consecutive-duplicates-string/
    '''
    # split and get length
    positions_to_remove = []
    S=S.split()
    # Traversing all combinations of string
    for i in range(len(S)):
        n_list = range(1,int(len(S[i:])/2)+1)
        #print("position:",i)
        for n in n_list:
            the_current = " ".join(S[i:i+n])
            #print(the_current)
            next_start = i+n
            next_end = next_start+n
            the_next = " ".join(S[next_start:next_end])
            #print(the_next)
            if the_next == the_current:
                #print("!!!repetitive here!!!")
                positions_to_remove = positions_to_remove + list(range(next_start,next_end))

    new_str=""
    for i in range(len(S)):
        if i not in positions_to_remove:
            new_str = new_str + " " + S[i]

    return new_str.strip()

def clean_df(df):
    '''
    Wrap up postmoderations above.
    Args:
        df -> pandas DataFrame
    Returns:
        pandas DataFrame.
    '''
    df.subjects = df.subjects.apply(lambda x: remove_repetitive(remove_non_english(to_string(x),change_is=False)))
    df.relations = df.relations.apply(lambda x: remove_repetitive(remove_non_english(to_string(x),change_is=True)))
    df.objects = df.objects.apply(lambda x: remove_repetitive(remove_non_english(to_string(x),change_is=False)))
    return df



'''
Main
'''
def export_triplets_with_meta(text):
    '''
    [Global Main]
    Args:
        text -> string
    Returns:
        A triplets DataFrame with metadata like AEO and Object-based Clusters.
    '''
    # clean
    cleaned_text = clean_text(text)
    # convert
    df = text2df(cleaned_text)
    # initiate
    tri_df = pd.DataFrame()
    # add meta, AEO
    for i in range(len(df)):
        sent = df.Texts[i]
        chunk_df = chunk_tri(sent)

        if len(chunk_df) == 0:
            tri_df_i = pd.DataFrame({
                "subjects": [[]],"relations": [[]],"objects": [[]], "texts": [sent], "need_curation": [False], "AEO_cat": [""], "sent_num": [i]
            })
            tri_df = tri_df.append(tri_df_i)
        else:
            tri_df_i = chunk_df
            tri_df_i = add_aeo_df(chunk_df)
            tri_df_i["sent_num"] = [i] * len(tri_df_i)
            tri_df = tri_df.append(tri_df_i)
    # clean df with meta
    tri_df = clean_df(tri_df)
    #print(tri_df.head())

    # add more meta, object-based clustering
    try:
        tri_df["objects_tokens"] = tri_df.objects.apply(lambda x: nlp(str(x)))
        tri_df["relations_tokens"] = tri_df.relations.apply(lambda x: nlp(str(x)))
        tri_df = add_obj_based_cluster(tri_df)
        return tri_df

    except:
        print("There is NO triplet extracted in this text.")


'''
Test Cases
'''
# print(export_triplets_with_meta(" But they knew what's good business for them, because our phone was disconnected."))
# print(export_triplets_with_meta("These guys were very rough and very bad."))
# print(export_triplets_with_meta("They used to beat you up, and scream, and holler, and push you"))
# print(export_triplets_with_meta("I am a German citizen."))
