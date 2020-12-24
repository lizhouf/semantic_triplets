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
Patterns and Lists
'''
vc_pattern = r'<VERB>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<VERB>+'
nc_pattern_noun = r'<DET>*<ADJ>*<NUM>*<PRON>?<PROPN>?<NOUN>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>*<NOUN>+'
nc_pattern_pron = r'<DET>*<ADJ>*<NUM>*<PRON>?<PROPN>?<NOUN>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>*<PRON>+'
nc_pattern_propn = r'<DET>*<ADJ>*<NUM>?<PRON>?<PROPN>?<NOUN>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>*<PROPN>+'
nc_pattern_adj = r'<DET>*<ADJ>?<NUM>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>+'
nc_pattern_num = r'<DET>*<ADJ>?<NUM>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<NUM>+'

speaking_verbs = ["said","say","told","tell"]
sub_keep_list = ["i","we","they","he","she"]

'''
Helpers for pre-moderation
'''

# helper: check if span contained in other span
def in_range(list1,list2):
    if (list1.start > list2.start and list1.end <= list2.end) or (list1.start >= list2.start and list1.end < list2.end):
        return True
    else:
        return False  # note: if list1==list2, still False


# helper: sort textacy object by start
def sort_by_start(the_list):
    start_list = [ele.start for ele in the_list]
    df = pd.DataFrame({"ele": the_list,"start": start_list})
    df = df.sort_values('start').reset_index(drop=True)
    return df.ele


# helper: merge spans
def merge_spans(span1,span2):
    with doc.retokenize() as retokenizer:
        return retokenizer.merge(doc[span1.start: span2.end])


# helper: find conj and comma position
def conj_punct_pos_list(doc):
    the_list = []
    for token in doc:
        if token.pos_ == "CCONJ" or token.pos_ == "PUNCT":
            the_list.append(token.i)
    return the_list

# helper: simplify subjects
# return a string
# use in subject
def simple_sub(doc):
    new_doc = ""
    for token in doc:
        if (token.text.lower() in sub_keep_list):
            new_doc = new_doc + str(token.text)+" "
            return new_doc[:-1]
    return doc

# helper: simplify objects which also serve as the subject of a clause
# use in subject
def simple_obj(doc):
    new_doc = ""
    for token in doc:
        print(token.dep_ )
        if (token.dep_ not in ['nsubj','nsubjpass','csubj','csubjpass']):
            new_doc = new_doc + str(token.text)+" "
    return new_doc[:-1]

'''
Extraction
'''
def all_possible_chunks(doc):
    '''
    extract all possible chunks
    :param sent: the input sentence
    :return: verb chunk and noun chunk lists
    '''

    ### Extract

    vc_list = []
    nc_list = [] # every chunk other than verb chunk is regarded as a noun chunk (nc)

    # verb chunk
    lists = textacy.extract.pos_regex_matches(doc, vc_pattern)
    for lista in lists:
        vc_list.append(lista)

    # noun chunk
    lists = textacy.extract.pos_regex_matches(doc, nc_pattern_noun)
    for lista in lists:
        nc_list.append(lista)

    # pron chunk
    lists = textacy.extract.pos_regex_matches(doc, nc_pattern_pron)
    for lista in lists:
        nc_list.append(lista)

    # propn chunk
    lists = textacy.extract.pos_regex_matches(doc, nc_pattern_propn)
    for lista in lists:
        nc_list.append(lista)

    # adj chunk
    lists = textacy.extract.pos_regex_matches(doc, nc_pattern_adj)
    for lista in lists:
        nc_list.append(lista)

    # num chunk
    lists = textacy.extract.pos_regex_matches(doc, nc_pattern_num)
    for lista in lists:
        nc_list.append(lista)

    return vc_list, nc_list

def filter_sort_chunks_df(vc_list,nc_list):
    # filter vc: remove if span contained another span
    # and append consecutive vc
    vc_final = []
    for list1 in vc_list:
        is_in_range = False
        for list2 in vc_list:
            if in_range(list1, list2):
                is_in_range = True
        if is_in_range:
            continue
        else:
            vc_final.append(list1)

    # filter nc: remove if span contained another span
    nc_final = []
    for list1 in nc_list:
        is_in_range = False
        for list2 in nc_list:
            if in_range(list1, list2):
                is_in_range = True
        if not is_in_range:
            nc_final.append(list1)

    # sort vc and nc by the order of occur
    vc_final = sort_by_start(vc_final)
    nc_final = sort_by_start(nc_final)

    positions_list = [list(range(int(ele.start),int(ele.end))) for ele in vc_final] + [
        list(range(int(ele.start),int(ele.end))) for ele in nc_final]
    df = pd.DataFrame({"ele": [vc for vc in vc_final] + [nc for nc in nc_final],
                       "position": positions_list,
                       "start": [i[0] for i in positions_list],
                       "category": ["V"] * len(vc_final) + ["N"] * len(nc_final)})
    df = df.sort_values('start').reset_index(drop=True)
    df = df.drop("start",axis=1)

    return df

def combine_chunks_in_df(df,doc):
    ### Combine

    # add subcategory to the df
    sub_cat_list = []
    for i in range(len(df)):
        this_cat = df.category[i]
        sub_cat = ""
        # vp means verb phrase and np means noun phrase

        if this_cat=="V":
            if any(j in [token.text.lower() for token in df.ele[i]] for j in speaking_verbs):
                sub_cat = "speaking_vp"
            elif df.ele[i][0].pos_ == "PART": #any(j in [token.pos_ for token in df.ele[i]] for j in ["PART"]) and : # main need improve!
                sub_cat = "infinitive_vp"
            else:
                sub_cat = "normal_vp"

        elif this_cat=="N":
            if df.ele[i][0].pos_ == "ADP":# any(j in [token.pos_ for token in df.ele[i]] for j in ["ADP"]):
                sub_cat = "indirect_np"
            elif not any(j in [token.pos_ for token in df.ele[i]] for j in ["NOUN","PRON","PROPN"]):
                sub_cat = "adj_num_np"
            else:
                sub_cat = "normal_np"
        sub_cat_list.append(sub_cat)
    df["subcat"] = sub_cat_list

    #print(df)

    # start choosing...

    ele_list = []
    pos_list = []
    cat_list = []
    subcat_list = []
    should_jump = False

    for i in range(len(df)):
        if should_jump == True:
            should_jump = False
            continue
        this_subcat = df.subcat[i]

        # fix - infinitive_vp
        try:
            if this_subcat == "infinitive_vp" and df.category[i-1]=="N" and df.category[i+1]=="N":
                ele_list[-1] = doc[df.position[i-1][0]:df.position[i+1][-1]+1]
                pos_list[-1] = df.position[i-1]+df.position[i]+df.position[i+1]
                should_jump = True
                continue
        except:
            0

        # fix - indirect_np
        try:
            if this_subcat == "indirect_np" and df.category[i-1]=="N":
                ele_list[-1] = doc[df.position[i-1][0]:df.position[i][-1]+1]
                pos_list[-1] = df.position[i-1]+df.position[i]
                continue
        except:
            0

        # all other situations
        ele_list.append(df.ele[i])
        pos_list.append(df.position[i])
        cat_list.append(df.category[i])
        subcat_list.append(df.subcat[i])


    df_combined = pd.DataFrame({"ele":ele_list,
                                "position":pos_list,
                                "category":cat_list,
                                "subcat":subcat_list,
                                "start":[i[0] for i in pos_list],
                                "end":[i[-1] for i in pos_list]})

    return df_combined

def order_chunks_in_df(df_combined,doc):
    ### Order
    # using conj and punct to divide the chunk list to several areas
    cp_list = conj_punct_pos_list(doc)
    triplets_sub = []
    triplets_rel = []
    triplets_obj = []

    # add 0 and sentence len-1 to this list
    cp_list = [0] + cp_list + [len(doc) - 1]
    cp_list = sorted(list(set(cp_list)))
    # print(cp_list)

    for i in range(len(cp_list) - 1):
        the_start = cp_list[i]
        the_end = cp_list[i + 1]
        area_df = df_combined[(df_combined["start"] >= the_start) & (df_combined["end"] <= the_end)].reset_index(drop=True)
        # print(area_df)
        if len(area_df) < 1:
            continue
        str_area_cat = "".join(list(area_df.category))
        # print(str_area_cat)
        # if contain speaking VP, continue
        # if "speaking_vp" in list(area_df.subcat):
        #     continue
        # if any area as NVN, it is a standing alone triplet
        if str_area_cat == "NVN":
            triplets_sub.append(simple_sub(area_df.ele[0]))
            triplets_rel.append(area_df.ele[1])
            triplets_obj.append(area_df.ele[2])
        # if only VN, use the subject NC from the previous triplet (conjunction)
        elif str_area_cat == "VN":
            try:
                triplets_sub.append(triplets_sub[-1])
                triplets_rel.append(area_df.ele[0])
                triplets_obj.append(area_df.ele[1])
            except:
                0
        # if NVNN (some indirect objects are not recognized as N)
        # we can think about REMOVING the indirect object category?
        elif "NVNN" in str_area_cat and str_area_cat.count("V") == 1:
            # find the position of V
            v_pos = area_df.index[area_df.category == "V"][0]  # should only has len 1
            # append
            try:
                triplets_sub.append(simple_sub(area_df.ele[v_pos - 1]))
                triplets_rel.append(area_df.ele[v_pos])
                triplets_obj.append(str(area_df.ele[v_pos + 1]) + " " + str(area_df.ele[v_pos + 2]))
            except:
                0

        # if NVNVN (should be not appended - so the last VN is an indirect object)
        # find the position of the first V
        elif "NVNVN" in str_area_cat:
            v_pos = area_df.index[area_df.category == "V"][0]  # should has len>=1
            # append
            try:
                triplets_sub.append(simple_sub(area_df.ele[v_pos - 1]))
                triplets_rel.append(area_df.ele[v_pos])
                triplets_obj.append(
                    str(area_df.ele[v_pos + 1]) + " " + str(area_df.ele[v_pos + 2]) + " " + str(area_df.ele[v_pos + 3]))
            except:
                0
        # if NVNV (maybe related to a clause)
        # we keep the first V
        elif "NVNV" in str_area_cat:
            # find the position of the first V
            v_pos = area_df.index[area_df.category == "V"][0]  # should has len>=1
            # append
            try:
                triplets_sub.append(simple_sub(area_df.ele[v_pos - 1]))
                triplets_rel.append(area_df.ele[v_pos])
                triplets_obj.append(simple_obj(area_df.ele[v_pos + 1]))
            except:
                0
        # otherwise, if more than NVN, only keep the NVN
        elif "NVN" in str_area_cat and str_area_cat.count("V") == 1:
            # find the position of V
            v_pos = area_df.index[area_df.category == "V"][0]  # should only has len 1
            # print(area_df)
            # print(v_pos)
            # append
            try:
                triplets_sub.append(simple_sub(area_df.ele[v_pos - 1]))
                triplets_rel.append(area_df.ele[v_pos])
                triplets_obj.append(area_df.ele[v_pos + 1])
            except:
                0
        # if all N's, append the whole chunk to the previous
        elif re.match(r"N{1,}",str_area_cat):
            try:
                triplets_obj[-1] = str(triplets_obj[-1]) + " " + " ".join([str(x) for x in area_df.ele])
            except:
                0
        # otherwise, if more than NVN, only keep the NVN
        elif "NVN" in str_area_cat and str_area_cat.count("V") == 1:
            # find the position of V
            v_pos = area_df.index[area_df.category == "V"][0]  # should only has len 1
            # append
            triplets_sub.append(area_df.ele[v_pos - 1])
            triplets_rel.append(area_df.ele[v_pos])
            triplets_obj.append(area_df.ele[v_pos + 1])

        # for all other situation, including NV, ignore

    # print(len(triplets_sub),len(triplets_rel),len(triplets_obj))

    df = pd.DataFrame({"subjects": [nlp(x) if type(x)==str else x for x in triplets_sub],
                       "relations": [nlp(x) if type(x)==str else x for x in triplets_rel],
                       "objects": [nlp(x) if type(x)==str else x for x in triplets_obj]})

    return df

def chunk_tri(sent):
    doc = textacy.make_spacy_doc(sent,lang='en_core_web_lg')
    vc_list,nc_list = all_possible_chunks(doc)
    df = filter_sort_chunks_df(vc_list,nc_list)
    df_combined = combine_chunks_in_df(df,doc)
    tri_df = order_chunks_in_df(df_combined,doc)
    tri_df["texts"] = [sent] * len(tri_df)
    return tri_df

# # test
# print(chunk_tri("I am Jewish."))