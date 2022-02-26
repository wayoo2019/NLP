#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:22:59 2022

@author: Yuan Wang
"""

import numpy as np
# import pandas as pd
import nltk
from nltk.corpus import brown
from nltk import ngrams
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors

lemmatizer = WordNetLemmatizer()
# use mcolors.CSS4_COLORS to define color set
colors_set = list(mcolors.CSS4_COLORS) 


# Define prediction function
def predict(color, objects):  #  color and objects from inout file
    # tokenize brown words
    tokens = list(brown.words()) 
    # create a dict for 2grams and corresponding frequency
    fdist = FreqDist(ngrams(tokens, 2))  

    # create candidates list for the prediction to refer
    candidates = []  #include the color+object pairs which color is a letter
    predicts = [] #final prediction list 
    others = [] #potential color+object pairs for test2
    freq = 0 # to count frequency of pairs in brown words for test 2
    
    # test 1: predict 5 words
    for i in range(len(objects)):  
        # k[0] is color, k[1] is object, v is frequency from fdist dict
        for k, v in fdist.items():  
            # if k[1], the fdist object is euqual to a input object 
            if k[1] == objects[i]: 
                #and k[0] is in the colors_set and color[i] (a letter) from input is equal to the first letter of k[0] from fdist
                if k[0] in colors_set and color[i] == k[0].lower()[0]:
                    # add this k,v to candidates pool
                    candidates.append((k, v))  
                    # if not, just append the original input to the candidates pool, give frequency = 1
            else:  
                candidates.append(((color[i], objects[i]), 1))
                #create a dict for candidates for better calculation
        candidates_ = dict(candidates)  
        pairs = list(max(zip(candidates_.values(), candidates_.keys()))[1])


        # test2: input color does not in the Brown corpus
        # pair[0] is color[i] which does not in colors_set
        if pairs[0] not in colors_set:  
            #check in colors_set if there is any c that can find in fdist dict, and pairs[0] also in c
            for c in colors_set:  
                for k, v in fdist.items():
                    if pairs[0] in set(c) and k[0] == c:  
                        # add this k, v into others dict 
                        freq += v  
                others.append((c, freq))
                freq = 0 # reset value
            # create others dict for frequency selection
            others_ = dict(others)
            # retuen the keys with highest frquency, like use frequency to predict
            freq_k = max(zip(others_.values(), others_.keys()))[1] 
            #update in potential list
            pairs[0] = freq_k 

        predicts.append(' '.join(pairs))
        candidates.clear()


    return predicts


# define function for token and lemmatize each line 
def parse_line(line):  
    line_split = nltk.word_tokenize(line)
    lines = []
    for i in range(len(line_split)):
        lines.append(lemmatizer.lemmatize(line_split[i]))
    return lines



def main():
    import sys
    # define input file as f1, read line byline, extract color and objects from each line, convert them to 2 arrays
    with open(sys.argv[1]) as f1:

        vals = [parse_line(line) for line in f1]
        (color, objects) = ([v[0] for v in vals], [v[1] for v in vals])
        np.asarray(color)
        np.asarray(objects)
    
    # define output files as f2, write prediction result in different lines
    with open(sys.argv[2]) as f2:
        contents = f2.read()
        predicts = predict(color, objects)
        print(predicts)
        my_file = open((sys.argv[2]), "r+")
        for i in range(len(predicts)):
            my_file.write(str(predicts[i]) + "\n")
        my_file.close()


if __name__ == '__main__':
    main()