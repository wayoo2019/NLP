#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:49:01 2022

@author: yuan wang/ywang340
"""
##
#import packages

import sys
import pandas as pd
import numpy as np
import json
import jsontrips
import re
import nltk
import gensim.downloader as api
from gensim.models import Word2Vec
import itertools
from itertools import chain, combinations

## method 1
# method 1: word2vec using predefined model glove-wiki-gigaword-100 

word_vectors = api.load("glove-wiki-gigaword-100")


def word2vec_metric(word1, word2):
    similarity_ = word_vectors.similarity(word1, word2)
    return word1, word2, similarity_


## method 2:
# read json file for Trip lexicon

with open('lex-ont.json', 'r') as myfile:
    data = myfile.read()
lexicon_dict = json.loads(data)

# load Trip Ontology

ontology_dict = jsontrips.ontology()


# create ecursive function to get all the parents and compare
def parent_recursion(x, parents):
    try:
        y = ontology_dict[x]['parent']
        parents.append(y)
        y, count = parent_recursion(y, parents)
        return y, parents
    except:
        return y, parents


# method 2: Wu-Palmer metric on TRIPS ontology
def Wu_Palmer_metric(word1, word2):
    y1 = lexicon_dict[word1]['lf_parent']
    y2 = lexicon_dict[word2]['lf_parent']
    parents1 = []
    parents2 = []
    for i in range(len(y1)):
        root, parent = parent_recursion(y1[i], [y1[i]])
        parents1.append(parent)

    for i in range(len(y2)):
        root, parent = parent_recursion(y2[i], [y2[i]])
        parents2.append(parent)

    Depth_LCS = 0
    similarity_ =0
    for i in range(len(parents1)):
        for j in range(len(parents2)):
            common_root = list(set(parents1[i]).intersection(set(parents2[j])))
            if len(common_root) > Depth_LCS:
                Depth_LCS = len(common_root)
                Depth_word1 = len(parents1[i])
                Depth_word2 = len(parents2[j])
                similarity_ = 2 * Depth_LCS / (Depth_word1 + Depth_word2)
    return word1, word2, similarity_

## Method 3
# loading Brown corpus

brown_lemmatized = open("brown_lemmatized.txt", "r")
Brown_corpus = []
for string in brown_lemmatized:
    new_string = re.sub(r"\W+|_", " ", string)
    word_token = nltk.word_tokenize(new_string)
    Brown_corpus += word_token

# lexicon
with open('trips-brown_NV_overlap.txt') as f:
    vals = [nltk.word_tokenize(line) for line in f]
    words_lexicon = [v[0] for v in vals]


# metrhod 3: brown copus-vector

def word_vector(word1, word2):
    v1 = np.zeros(len(words_lexicon))
    v2 = np.zeros(len(words_lexicon))

    for i in range(len(Brown_corpus)):
        if word1 == Brown_corpus[i]:
            for j in range(i - 4, i + 5):
                if Brown_corpus[j] in words_lexicon and j != i:
                    k = words_lexicon.index(Brown_corpus[j])
                    v1[k] += 1

        if word2 == Brown_corpus[i]:
            for j in range(i - 4, i + 5):
                if Brown_corpus[j] in words_lexicon and j != i:
                    k = words_lexicon.index(Brown_corpus[j])
                    v2[k] += 1

        # cosine matrix
    Lv1 = np.sqrt(v1.dot(v1))
    Lv2 = np.sqrt(v2.dot(v2))

    similarity_ = v1.dot(v2) / (Lv1 * Lv2)

    return word1, word2, similarity_


## method 4:
# 4th novel method

brown_lemmatized = open("brown_lemmatized.txt", "r")
Brown_corpus2 = []

for string in brown_lemmatized:
    new_string = re.sub(r"\W+|_", " ", string)
    word_token =nltk.word_tokenize(new_string)
    Brown_corpus2.append(word_token)


def forth_technique(word1, word2):
    model = Word2Vec(sentences=Brown_corpus2, min_count=1)
    # model.save("word2vec.model")
    word_vectors2 = model.wv
    # word_vectors2.save("word2vec.wordvectors")
    similarity_ = word_vectors2.similarity(word1, word2)
    return word1, word2, similarity_


## Result part
# score file output

def predict(y):
    score = pd.DataFrame()
    tripleid = []
    words1 = []
    words2 = []
    sim1 = []
    sim2 = []
    sim3 = []
    sim4 = []
    for i in range(len(y)):
      for j in range(len(y[i])):
        word1, word2, similarity1 = word2vec_metric(y[i][j][0], y[i][j][1])
        word1, word2, similarity2 = Wu_Palmer_metric(y[i][j][0].upper(), y[i][j][1].upper())
        word1, word2, similarity3 = word_vector(y[i][j][0], y[i][j][1])
        word1, word2, similarity4 = forth_technique(y[i][j][0], y[i][j][1])
        tripleid.append(i)
        words1.append(word1)
        words2.append(word2)
        sim1.append(similarity1)
        sim2.append(similarity2)
        sim3.append(similarity3)
        sim4.append(similarity4)
    
    # words1, words2, sims
    score['tripleid'] = tripleid
    score['word1'] = words1
    score['word2'] = words2
    score['word2vec_score'] = sim1
    score['Wu_Palmer_score'] = sim2
    score['brown-vector_score'] = sim3
    score['4th_novel_score'] = sim4
    return score


# output file
def result(score):
    stranges1 = []
    stranges2 = []
    stranges3 = []
    stranges4 = []
    
    tripleid = []
    output = pd.DataFrame()
    for i in list(set(score['tripleid'].values)):
      group = score[score['tripleid']==i]
      all = list(chain(*(group[['word1', 'word2']].values.tolist())))
      high1 = list(chain(*(group[group['word2vec_score']==group['word2vec_score'].max()][['word1', 'word2']].values.tolist())))
      high2 = list(chain(*(group[group['Wu_Palmer_score']==group['Wu_Palmer_score'].max()][['word1', 'word2']].values.tolist())))
      high3 = list(chain(*(group[group['brown-vector_score']==group['brown-vector_score'].max()][['word1', 'word2']].values.tolist())))
      high4 = list(chain(*(group[group['4th_novel_score']==group['4th_novel_score'].max()][['word1', 'word2']].values.tolist())))
    
      str1 = list(set(all) - set(high1))
      str2 = list(set(all) - set(high2))
      str3 = list(set(all) - set(high3))
      str4 = list(set(all) - set(high4))  
      stranges1.append(str1)
      stranges2.append(str2)
      stranges3.append(str3)
      stranges4.append(str4)
      tripleid.append(i)
    
    output['tripleid'] = tripleid
    output['word2vec_score_choice'] = stranges1
    output['word2vec_score_choice'] = output['word2vec_score_choice'].str[0]
    output['Wu_Palmer_score_choice'] = stranges2
    output['Wu_Palmer_score_choice'] = output['Wu_Palmer_score_choice'].str[0]
    output['brown-vector_score_choice'] = stranges3
    output['brown-vector_score_choice'] = output['brown-vector_score_choice'].str[0]
    output['4th_novel_score_choice'] = stranges4
    output['4th_novel_score_choice'] = output['4th_novel_score_choice'].str[0]
    return output
 


##
def main():
    with open(sys.argv[1]) as f1:
        inputs = pd.read_csv(f1)
        x = inputs.to_string(header=False, index=False,index_names=False).split('\n')
        vals = [' '.join(ele.split()) for ele in x]
        y = []
        for i in range(len(vals)):
          line = nltk.word_tokenize(vals[i])
          x = list(itertools.combinations(line, 2))
          y.append(x)

    with open(sys.argv[3], 'w+',) as f3:
        score = predict(y)
        print(score)
        score.to_csv('score.csv', index=False)
        f3.close()

    with open(sys.argv[2], 'w+',) as f2:
        output = result(score)
        print(output)
        output.to_csv('output.csv', index=False)
        f2.close()



if __name__ == '__main__':
    main()