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
Patterns and Lists
'''
vc_pattern = r'<VERB>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<VERB>+'
nc_pattern_noun = r'<DET>*<ADJ>*<NUM>*<PRON>?<PROPN>?<NOUN>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>*<NOUN>+'
nc_pattern_pron = r'<DET>*<ADJ>*<NUM>*<PRON>?<PROPN>?<NOUN>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>*<PRON>+'
nc_pattern_propn = r'<DET>*<ADJ>*<NUM>?<PRON>?<PROPN>?<NOUN>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>*<PROPN>+'
nc_pattern_adj = r'<DET>*<ADJ>?<NUM>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<ADJ>+'
nc_pattern_num = r'<DET>*<ADJ>?<NUM>?<AUX>*<ADV>*<PART>*<ADP>*<DET>*<NUM>+'

#speaking_verbs = ["said","say","told","tell"]
sub_keep_list = ["i","we","they","he","she"]


'''
Helpers for pre-moderation
'''

def in_range(list1,list2):
    """
    Check if one Span is contained in other Span.
    Args:
        list1 -> spaCy Span
        list2 -> spaCy Span
    Returns:
        True if list1 is contained in list2
        False otherwise
    Notes:
        If list1 equals to list2, the function still returns False
    """
    if (list1.start > list2.start and list1.end <= list2.end) or (list1.start >= list2.start and list1.end < list2.end):
        return True
    else:
        return False  

def sort_by_start(the_list):
    """
    Sort a list textacy objects by start positions.
    Args:
        the_list1 -> A list of textacy objects
    Returns:
        A list of textacy objects
    """
    start_list = [ele.start for ele in the_list]
    df = pd.DataFrame({"ele": the_list,"start": start_list})
    df = df.sort_values('start').reset_index(drop=True)
    return df.ele

def merge_spans(span1,span2):
    """
    Merge two spaCy Spans.
    Args:
        list1 -> spaCy Span
        list2 -> spaCy Span
    Returns:
        A spaCy Span
    """
    with doc.retokenize() as retokenizer:
        return retokenizer.merge(doc[span1.start: span2.end])

def conj_punct_pos_list(doc):
    """
    In a spaCy Doc, find positions of tokens with "CCONJ" or "PUNCT" as POS.
    Args:
        doc -> spaCy Doc
    Returns:
        A list of numbers
    """
    the_list = []
    for token in doc:
        if token.pos_ == "CCONJ" or token.pos_ == "PUNCT":
            the_list.append(token.i)
    return the_list

def simple_sub(doc):
    """
    Simplifies subjects from a spaCy Doc to a string and only keeps a selected categories of subjects.
    Used for clean up Triplets' subjects.
    Args:
        doc -> spaCy Doc
    Returns:
        A string
    """
    new_doc = ""
    for token in doc:
        if (token.text.lower() in sub_keep_list):
            new_doc = new_doc + str(token.text)+" "
            return new_doc[:-1]
    return doc

def simple_obj(doc):
    """
    Simplifies objects from a spaCy Doc to a string and only keeps a selected categories of objects.
    Used for clean up Triplets' objects.
    Args:
        doc -> spaCy Doc
    Returns:
        A string
    """
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
    Extracts all possible chunks.
    Args:
        doc -> spaCy Doc
    Returns:
        Lists of verb chunks and noun chunk s
    '''

    vc_list = []
    nc_list = [] # every chunk other than verb chunk is in the larger category of noun chunk (nc)

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
    '''
    Filter and sort all chunks.
    Args:
        vc_list -> a list of spaCy Docs
        nc_list -> a list of spaCy Docs
    Returns:
        A df with sorted chunks and related metadata
    '''

    # filter vc: remove if span contained another span
    # and append consecutive vc (vc only, not for nc)
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
    #print(nc_final)

    positions_list = [list(range(int(ele.start),int(ele.end))) for ele in vc_final] + [
        list(range(int(ele.start),int(ele.end))) for ele in nc_final]
    df = pd.DataFrame({"ele": [vc for vc in vc_final] + [nc for nc in nc_final],
                       "position": positions_list,
                       "start": [i[0] for i in positions_list],
                       "end": [i[-1] for i in positions_list],
                       "category": ["V"] * len(vc_final) + ["N"] * len(nc_final)})
    df = df.sort_values('start').reset_index(drop=True)

    return df

def order_chunks_in_df(df_combined,doc):
    '''
    Order all chunks.
    Args:
        df_combined -> a df of chunks
        doc -> a spaCy Doc
    Returns:
        A df with sorted chunks as subjects, relations, and objects.
    '''

    ### segment the whole Doc into areas
    # using conj and punct to divide the chunk list to several areas
    cp_list = conj_punct_pos_list(doc)
    #print(cp_list)
    # add 0 and sentence len-1 to this list
    cp_list = [0] + cp_list + [len(doc) - 1]
    cp_list = sorted(list(set(cp_list)))
    # print(cp_list)

    ### Adjust chunks in each segment
    # Initiate lists for subjects, relations, and objects
    triplets_sub = []
    triplets_rel = []
    triplets_obj = []
    # Iterate the Doc and adjust chunks in each segment by N & V patterns
    for i in range(len(cp_list) - 1):
        the_start = cp_list[i]
        the_end = cp_list[i + 1]
        area_df = df_combined[(df_combined["start"] >= the_start) & (df_combined["end"] <= the_end)].reset_index(drop=True)
        # print(area_df)
        if len(area_df) < 1:
            continue
        str_area_cat = "".join(list(area_df.category))
        #print(str_area_cat)
        #print(area_df)
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
            v_pos = area_df.index[area_df.category == "V"][0]  # should have len>=1
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
            v_pos = area_df.index[area_df.category == "V"][0]  # should have len>=1
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
            v_pos = area_df.index[area_df.category == "V"][0]  # should only have len 1
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

def to_curate(doc, obj_doc,rel_doc):
    '''
    Decide if curation is needed.
    Args:
        doc -> spaCy doc
        obj_doc -> spaCy Doc
        rel_doc -> spaCy Doc
    Returns:
        bool, True for need curation and False otherwise.
    '''

    # for long sentences
    if len(obj_doc)+len(rel_doc)>25:
        return True

    # for speaking words, possibly quotations
    speaking_verbs = ["said","say","told","tell"]
    if any(j in [token.text.lower() for token in doc] for j in speaking_verbs):
        return True

    # for complicated objects with more than 3 noun-like objects
    # reference: https://www.english-grammar-revolution.com/list-of-pronouns.html
    people_pron = ["i","me","we","us","you","she","her","he","him","it","they","them",
                    "who","whom","whose","whomever",
                    "anybody","anyone","everybode","everyone",
                    "nobody","noone","somebody","someone",
                    "myself","ourselves","yourself","yourselves",
                    "himself","herself","itself","themselves"]
    noun_in_obj = [1 if token.text.lower() in people_pron else 0 for token in obj_doc]
    if sum(noun_in_obj)>=3:
        return True

    # otherwise
    return False

def to_remove(sub_doc):
    '''
    Decide if curation is needed.
    Args:
        sub_doc -> spaCy Doc
    Returns:
        bool, True this triplet need to be removed and False otherwise.
    '''
    # what and who cannot be in subjects
    wh_no_sub = ["what","who"]
    if any(j in [token.text.lower() for token in sub_doc] for j in wh_no_sub):
        return True

    # otherwise
    return False


def chunk_tri(sent):
    '''
    [Local Main Function]
    Export Triplets in the chunk format
    Args:
        sent -> a string
    Returns:
        A df with columns subjects, relations, objects, and texts
    '''
    doc = textacy.make_spacy_doc(sent,lang='en_core_web_lg')
    vc_list,nc_list = all_possible_chunks(doc)
    df = filter_sort_chunks_df(vc_list,nc_list)
    tri_df = order_chunks_in_df(df,doc)
    tri_df["texts"] = [sent] * len(tri_df)
    # curation & remove
    need_curation = []
    need_remove = []
    for i in range(len(tri_df)):
        need_curation.append(to_curate(doc,tri_df.objects[i],tri_df.relations[i]))
        need_remove.append(to_remove(tri_df.subjects[i]))
    tri_df["need_curation"] = need_curation
    tri_df["need_remove"] = need_remove
    tri_df = tri_df[tri_df.need_remove==False].reset_index(drop=True).drop(columns=["need_remove"])

    return tri_df

'''
Test Cases
'''
# print(chunk_tri(" But they knew what's good business for them, because our phone was disconnected."))
# print(chunk_tri("These guys were very rough and very bad."))
# print(chunk_tri("They used to beat you up, and scream, and holler, and push you"))
