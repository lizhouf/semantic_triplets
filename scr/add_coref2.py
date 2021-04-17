'''
Import
'''
import pandas as pd
import numpy as np
# NLP
import spacy
nlp = spacy.load('en_core_web_lg')
# Grouping Coref
import networkx as nx
import itertools
# word
import re
import inflect
inflect = inflect.engine()
# tri
from tri_main import *

# note: some of the functions,
# including raw2token_relation, generate_token_df, and extract_coref_df
# are partly converted from Wanxin Xie's earlier code
# the creators of some of the word lists are also marked

'''
Word Lists
'''
pron = ["i","me","we","us","you","she","her","he","him","it","they","them",
        "who","whom","whose","whomever",
        "anybody","anyone","everybody","everyone",
        "nobody","noone","somebody","someone",
        "myself","ourselves","yourself","yourselves",
        "himself","herself","itself","themselves"] # Leo
priority_he_she = ["mother", "father", "sister", "brother", "aunt", "uncle", "cousin", "niece",
                   "nephew", "grandmother", "grandfather", "child", "son", "daughter",
                   "friend", "neighbor", "doctor", "physician", "teacher", "policeman",
                   "police", "soldier", "officer", "commander", "rabbi", "priest", "Nazi", "German", "Jew"] # Todd
priority_they = ["Jews", "Nazis", "soldiers", "SS", "officers", "Americans", "Russians", "Germans", "Hungarians", 
                 "Christians", "family", "Gestapo", "grandparents", "parents", "sons", "daughters"] # Todd
not_pelace_list = ["one","man"]
abb_list = ["t","re","m","s","d","clock"] # Leo
white_list_check_similar = ["a","an","the","this","that"] # Leo

'''
From ann and txt to Tokens in df
'''

def raw2token_relation(this_txt,this_ann):
	# ANN
	Lines = this_ann

	# Split lines to two groups for MENTIONS and COREF
	count = 0
	Token = []
	Relation = []

	for line in Lines: 
	    l = line.strip()
	    if l[0] == 'T':
	        Token.append(l.split('\t'))
	    else:
	        Relation.append(l.split('\t'))

	# TXT
	Lines2 = this_txt
	words = Lines2[0].split()
	final = []
	count = 0
	start = 0
	end=0
	for i in words:
	    temp = []
	    temp.append(i)
	    end = start + len(i)
	    temp.append(count)
	    temp.append(start)
	    temp.append(end)
	    start = end + 1
	    count += 1
	    final.append(temp)

	return Token,Relation,final

def generate_token_df(Token,Relation,final):
	# Mentions
	Mentions = []

	for tok in Token:
	    start_idx = 0
	    end_idx = 0
	    result = []
	    result.append('MENTION')
	    result.append(tok[0])
	    for k in range(len(final)):
	        if str(final[k][2]) == str(tok[1][1]):
	            start_idx = k
	            break
	    for j in range(len(final)):
	        if str(final[j][3]) == str(tok[1][2]):
	            end_idx = j
	            break
	    result.append(0)
	    result.append(start_idx)
	    result.append(0)
	    result.append(end_idx)
	    result.append(tok[-1])
	    result.append(tok[1][0])
	    
	    #print(tok[-1])
	    #print(len(tok[-1].split()))
	    if len(tok[-1].split()) == 1:
	        if str(tok[-1]).lower() in pron:
	            result.append("PRON")
	        else:
	            word = tok[-1].split()
	            doc = nlp(str(word))
	            pos = []
	            for token in doc:
	                pos.append(token.pos_)
	            if "PROPN" in pos:
	                result.append("PROP")
	            else:
	                result.append("NOM")    

	            
	    else:
	        doc = nlp(str(tok[-1]))
	        pos = []
	        for token in doc:
	            pos.append(token.pos_)
	        if "PROPN" in pos:
	            result.append("PROP")
	        else:
	            result.append("NOM")
	        

	    Mentions.append(result)

	# label, text, pos, start, end
	this_token = [x[1] for x in Mentions]
	this_text = [x[6] for x in Mentions]
	this_pos = [x[8] for x in Mentions]
	this_label = [x[1].split()[0] for x in Token]
	this_start = [x[1].split()[1] for x in Token]
	this_end = [x[1].split()[2] for x in Token]
	token_df = pd.DataFrame({"token":this_token,
	                         "text":this_text,
	                         "pos":this_pos,
	                         "label":this_label,
	                         "start":this_start,
	                         "end":this_end})
	return token_df

'''
Replace
'''

# helper
def has_word(old_str,new_str): # if old_str is in new_str
    single_word_pattern=re.compile(r'\b%s\b' % old_str, re.I)
    if re.search(single_word_pattern, new_str):
    	return True
    else:
    	return False

def find_proper_replace(sub_df):
    text_to_replace = ""
    
    # if contradictory, don't change
    has_contra = False
    has_he_she = False
    has_they = False
    has_priority_he_she = False
    has_priority_they = False
    for i in range(len(sub_df)):
        if any(has_word(to_check_word,sub_df.text[i]) for to_check_word in ["he","He","she","She"]):
            has_he_she = True
        if any(has_word(to_check_word,sub_df.text[i]) for to_check_word in ["they","They"]):
            has_they = True
        if any(has_word(to_check_word,sub_df.text[i]) for to_check_word in priority_he_she):
            has_priority_he_she = True
        if any(has_word(to_check_word,sub_df.text[i]) for to_check_word in priority_they):
            has_priority_they = True
    # detect if has contrast
    if has_he_she and has_they:
        has_contra =True
    if has_he_she and has_priority_they:
        has_contra =True
    if has_they and has_priority_he_she:
        has_contra =True
    if has_priority_he_she and has_priority_they:
        has_contra =True
    if has_contra:
        return text_to_replace # return empty
    
    # find priority lists
    for i in range(len(sub_df)):
        if any(has_word(to_check_word,sub_df.text[i]) for to_check_word in ["he","He","she","She"]):
            for coref_text in sub_df.text:
                if any(has_word(more_check_word,coref_text) for more_check_word in priority_he_she) and coref_text not in not_pelace_list:
                    return coref_text
        # TODO experiment: try to loose the rules for "they"
        elif any(has_word(to_check_word,sub_df.text[i]) for to_check_word in ["they","They"]):
            for coref_text in sub_df.text:
                if any(has_word(more_check_word,coref_text) for more_check_word in priority_they) and coref_text not in not_pelace_list:
                    return coref_text
    
    # if not in priority, find the earliest one
    for i in range(len(sub_df)):      
        if sub_df.pos[i] !="PRON":
            return sub_df.text[i]

    return text_to_replace

def add_token_quant(token):
	quant = "single" # by observing the results, default is single
	inflect_token = inflect.singular_noun(token) 
	if inflect_token:
		if inflect_token==token:
			quant = "single"
		else:
			quant = "plural"
	else: # inflect not detectable
		if any(has_word(to_check_word,token) for to_check_word in priority_he_she):
			quant = "single"
		elif any(has_word(to_check_word,token) for to_check_word in priority_they):
			quant = "plural"

	return quant



def extract_coref_df(Relation,token_df):
	# Extract coref pairs
	count = 0
	L = []
	for r in Relation:
	    L.append((r[1].split()[1].split(':')[1], r[1].split()[2].split(':')[1]))

	# Grouping coref
	G=nx.from_edgelist(L)
	l=list(nx.connected_components(G))
	mapdict={z:x for x, y in enumerate(l) for z in y }
	newlist=[ x+(mapdict[x[0]],)for  x in L]
	newlist=sorted(newlist,key=lambda x : x[2])
	coref_idx_list=[list(y) for x , y in itertools.groupby(newlist,key=lambda x : x[2])]

	# Create dictionary with assigned index
	coref_dict = dict()
	for i in range(len(coref_idx_list)):
	    temp = []
	    for j in range(len(coref_idx_list[i])):
	        temp.append(coref_idx_list[i][j][0])
	        temp.append(coref_idx_list[i][j][1])
	    temp = list(set(temp))
	    coref_dict[i]=temp

	final_corefdict = dict()
	for key, value in coref_dict.items():
	    for i in value:
	        final_corefdict[i] = key

	# add coref column to the token_df
	# and find the proper replacement
	token_df["coref_num"] = [-1]*len(token_df) # -1 means no coref
	token_df["coref_text"] = token_df.text.copy()
	for key in range(len(coref_dict)):
	    token_list=coref_dict[key]

	    for token in token_list:
	        # change the coref token
	        token_df.loc[token_df.token == token, 'coref_num'] = key
	        # find the proper replacement
	    sub_df = token_df[token_df.coref_num==key].reset_index(drop=True)
	    text_to_replace = find_proper_replace(sub_df)
	    
	    if text_to_replace!="":
	        token_df.loc[token_df.token.isin(token_list), 'coref_text'] = text_to_replace

	# add token quantity
	token_df["text_quant"] = token_df.text.apply(lambda x: add_token_quant(x))
	token_df["coref_text_quant"] = token_df.coref_text.apply(lambda x: add_token_quant(x))

	return token_df
	
'''
Pre-process
'''
def tokens2str(text):
	tokens = text.split(" ")
	full_str = ""
	for i in range(len(tokens)):
		if i!=0: # not the first token
			# if the token does NOT begin with alpha 
			# or in abb list
			try:
				if (not tokens[i][0].isalpha()) or (tokens[i] in abb_list): 
					full_str = full_str + tokens[i] # directly append
				else: 
					full_str = full_str + " " + tokens[i] # append with a space		
			except: # length is 0
				0 # do nothing
		else:
			full_str = full_str + tokens[i] # the first token
	return full_str

def clean_text(text):
	# the text_org column makes sure that positions are correct
	# the first step below, then simply put seperated punctuations together
	# append
	full_str = tokens2str(text)
	# periods
	cleaned = full_str.replace('...',' ... ')
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

def text2sentences(this_txt): 
	# first split
	sentence_pattern = re.compile(r"((?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s)") # with capturing
	sentences = re.split(sentence_pattern,this_txt)	
	# then clean
	sentences_clean = [clean_text(sent) for sent in sentences]	
	return sentences, sentences_clean

'''
Post Replacement
'''
def find_speaker(this_start,this_end,this_meta):
	this_meta.starting_index = this_meta.starting_index.apply(lambda x: int(x))
	this_meta.ending_index = this_meta.ending_index.apply(lambda x: int(x))
	for i in range(len(this_meta)):
		if this_meta.starting_index[i]<=this_start and this_meta.ending_index[i]>=this_end:
			#print(this_meta.speaker[i])
			return str(this_meta.speaker[i])

def replace_subject(to_replace_token,replace_by_token,current_subject,current_object): # single word replacement for subject
    single_word_pattern = re.compile(r'\b%s\b' % to_replace_token, re.I)
    # Don’t change the subject/object coreference to make them the same
    current_object_list_clean = [word.lower() for word in current_object.split(" ") if word.lower() not in white_list_check_similar ]
    if any(has_word(to_check_word.lower(),replace_by_token.lower()) for to_check_word in current_object_list_clean):
        return current_subject
    else:
        replace_candidate = re.sub(single_word_pattern, replace_by_token, current_subject)
        return replace_candidate

def replace_object(to_replace_token,replace_by_token,current_subject,current_object): # single word replacement for object
    single_word_pattern = re.compile(r'\b%s\b' % to_replace_token, re.I)
    current_subject_list_clean = [word.lower() for word in current_subject.split(" ") if word.lower() not in white_list_check_similar ]
    if any(has_word(to_check_word.lower(),replace_by_token.lower()) for to_check_word in current_subject_list_clean):
        return current_object
    else:
        replace_candidate = re.sub(single_word_pattern, replace_by_token, current_object)
        return replace_candidate

def replace_df_with_coref(tri_df, token_df):
	### Apply the tokens' results to the whole df
	### replace to a corefed version
	# texts_og, start, end
	# subjects, relations, objects, texts

	# initiate
	tri_df["subjects_coref"] = list(tri_df.subjects)
	tri_df["objects_coref"] = list(tri_df.objects)
	# print(tri_df.head()) # REMOVE

	# replace
	for i in range(len(token_df)):
		if (token_df.coref_num[i]!=-1 and # has coref
		token_df.pos[i]=="PRON" and # only change pronouns
		token_df.text_quant[i] == token_df.coref_text_quant[i]): # singular and plural consist - only wrong: both "", i.e. unknown
			# if quant unknown
			need_check = False
			# check for they, if got replaced
			if token_df.text[i].lower()=="they" and token_df.text[i] != token_df.coref_text[i]:
				need_check = True
			# adjust start and end
			this_start = int(token_df.start[i])
			this_end = int(token_df.end[i])
			# replace
			for j in range(len(tri_df)):
				df_start_j = int(tri_df.starts[j])
				df_end_j = int(tri_df.ends[j])
				# the line of df includes token
				if df_start_j<=this_start and df_end_j>=this_end:
					if need_check: # if not, don't change - can't simplify!
						tri_df.need_curation[j] = True
					current_subject = tri_df.subjects_coref[j]
					current_object = tri_df.objects_coref[j]
					to_replace_token = token_df.text[i]
					replace_by_token = token_df.coref_text[i]
					# replace subjects
					try: 
						tri_df.subjects_coref[j] = replace_subject(to_replace_token,replace_by_token,current_subject,current_object)
					except:
						0
					# replace objects
					try: 
						tri_df.objects_coref[j] = replace_object(to_replace_token,replace_by_token,current_subject,current_object)
					except:
						0

	return tri_df

'''
Main
'''
# data
data_path = "/Users/lizhoufan/Dropbox/HGSDLab/Import_Visualize_Annotations/data/V07/"
# file_names = ["Boder_56_Abraham_Kimmelmann_en_cleaned",
# 			"Shoah_8_cleaned",
# 			"Boder_31_Henja_Frydman_en_cleaned",
# 			"fortunoff_1_cleaned"
# 			]
file_names = ["Boder_56_Abraham_Kimmelmann_en_cleaned",
			]

for name in file_names:
	print("Starting "+name)

	# read in files
	this_txt = open(data_path+name+'.txt', 'r').readlines()
	this_ann = open(data_path+name+'.ann', 'r').readlines()
	this_meta = pd.read_csv(data_path+name+'.csv')

	# tokens and relations to df with coref
	Token, Relation, final = raw2token_relation(this_txt,this_ann)
	token_df = generate_token_df(Token,Relation,final)
	token_df_with_coref = extract_coref_df(Relation,token_df)
	token_df_with_coref.to_excel(data_path+name.split("_")[0]+"_"+name.split("_")[1]+'_tokens_with_coref_v07-2.xlsx',index=False)

	# generate corefed df
	# by replace text with coref
	sentences, sentences_clean = text2sentences(this_txt[0])
	#pd.DataFrame({"og":sentences,"new":sentences_clean}).to_excel(data_path+name.split("_")[0]+"_"+name.split("_")[1]+'_og_new_sentences.xlsx',index=False)
	tri_df = pd.DataFrame()
	end_in_df = -1
	for i in range(len(sentences_clean)):
		this_df = export_triplets_with_meta(sentences_clean[i])
		this_df["texts_og"] = sentences[i] # sentences and sentences_clean should be of the same length
		this_start = end_in_df+1
		this_end = this_start+len(sentences[i])-1
		end_in_df = this_end
		this_df["starts"] = [int(this_start)]*len(this_df)
		this_df["ends"] = [int(this_end)]*len(this_df)
		this_speaker = find_speaker(this_start,this_end,this_meta)
		this_df["speakers"] = [this_speaker]*len(this_df) ### TODO: need to change to 0 and 1 for speaker and interviewer later
		tri_df = tri_df.append(this_df)
	tri_df = tri_df.reset_index(drop=True)

	# deal with tri_df
	tri_df_coref = replace_df_with_coref(tri_df,token_df)
	tri_df_coref.to_excel(data_path+name.split("_")[0]+"_"+name.split("_")[1]+'_tri_with_coref_v07-2.xlsx',index=False)


