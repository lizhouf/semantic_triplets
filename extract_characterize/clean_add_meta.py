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
# add others
from add_aeo import add_aeo_df
from add_pvo import add_pvo_df
from tri_chunk_extract import chunk_tri


'''
Helper functions for cleaning and add meta
'''

def clean_text(text):
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


def wrangle_text(df):  # wrangle experiment
    df.text = df.text.apply(lambda x: clean_text(str(x) + "."
                                                 if (str(x)[-1] != "." and str(x)[-1] != "!" and str(x)[-1] != "?")
                                                 else clean_text(str(x))))  # make sure every cell is a string
    df.prt_key = df.prt_key.apply(lambda x: ast.literal_eval(x) if type(x) == str else [])
    print(df.head())


    interviewee_full = ' '.join(df['text'].tolist())

    # get rid of - and things in between
    # get rid of things in between ""

    sentence_pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    interviewee_list = re.split(sentence_pattern,interviewee_full)

    # df = pd.DataFrame({"Texts":interviewee_list,"Text_segments":interviewee_segment})
    tri_df = pd.DataFrame({"Texts": interviewee_list})
    tri_df = tri_df[tri_df.Texts != ""].reset_index(drop=True)

    # need to maintain the prt_key/page_key
    def find_seg_key(sent,df):
        seg_key = []
        for i in range(len(df)):
            if sent in df.text[i]:
                seg_key = df.prt_key[i]
                break
        return seg_key

    def find_kwd_key(sent,df):
        kwd_key = []
        for i in range(len(df)):
            if sent in df.text[i]:
                kwd_key = df.kwd_key[i]
                break
        return kwd_key

    tri_df["seg_key"] = tri_df.Texts.apply(lambda x: find_seg_key(x,df))
    tri_df["kwd_key"] = tri_df.Texts.apply(lambda x: find_kwd_key(x,df))

    return tri_df



'''
lists
'''

# metadata
cat_list = [
    "Time and place",
    "Organizations",
    "Movement",
    "Economics",
    "People",
    "Still and moving images",
    "World histories",
    "Discrimination responses",
    "Feelings and thoughts",
    "Post-conflict",
    "Captivity",
    "Religion and philosophy",
    "Forced labor",
    "Daily life",
    "Culture",
    "Refugee experiences",
    "Mistreatment and death",
    "Discrimination",
    "Health",
    "Government",
    "Liberation",
    "Politics"]
cat_list = [cat.lower() for cat in cat_list]

'''
Add meta
'''


def tri_export(df,Corpus_tag,File_num):  # ,name,save_path
    df = wrangle_text(df)
    print(df.head())
    df.seg_key = df.seg_key.apply(lambda y: [x.lower() for x in y])

    # add Index columns
    for cat_word in cat_list:
        df[cat_word] = df.seg_key.apply(lambda x: 1 if cat_word in x else 0)

    tri_df = pd.DataFrame()

    print("------- adding SVO & AEO_cat")

    ### add Triplets, AEO, and Meta
    for i in range(len(df)):
        sent = df.Texts[i]
        chunk_df = chunk_tri(sent)

        if len(chunk_df) == 0:
            tri_df_i = pd.DataFrame({
                "subjects": [[]],"relations": [[]],"objects": [[]], "texts": [sent], "AEO_cat": [""], "sent_num": [i]
            })
            for cat_word in cat_list:
                tri_df_i[cat_word.replace(" ","_").replace("-","_")] = [0]
            tri_df_i["kwd_key"] = [df.kwd_key[i]] * len(tri_df_i)
            tri_df = tri_df.append(tri_df_i)
        else:
            tri_df_i = chunk_df
            tri_df_i = add_aeo_df(chunk_df)
            tri_df_i["sent_num"] = [i] * len(tri_df_i)
            for cat_word in cat_list:
                tri_df_i[cat_word.replace(" ","_").replace("-","_")] = [df[cat_word][i]] * len(tri_df_i)
            tri_df_i["kwd_key"] = [df.kwd_key[i]] * len(tri_df_i)
            tri_df = tri_df.append(tri_df_i)



    # TODO: add coref
    # tri_df = add_coref(tri_df)

    # TODO: human select correct coref (not strict)
    # note: join the slection column to keep Spacy objects

    ### add PVO_cat
    # TODO: consider the coref
    tri_df = add_pvo_df(tri_df)

    ### add other meta
    tri_df["Corpus"] = [Corpus_tag] * len(tri_df)
    tri_df["IntCode"] = [File_num] * len(tri_df)

    # add human selection
    # tri_df["is_select"] = tri_df.subjects.apply(lambda x: 0 if x==[] else 1) # may need to manually add this...
    # note: join the slection column

    # export - if only keep has AEO, use the below
    tri_df = tri_df[tri_df.AEO_cat != ""].reset_index(drop=True)
    # tri_df.to_csv(save_path+name+".csv",index=False)

    # print(tri_df.columns)
    # print("META",tri_df.head())

    return tri_df.reset_index(drop=True)
