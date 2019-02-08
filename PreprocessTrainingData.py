#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
import string
import numpy as np
from ExportTrainingDataWithEntities import list_as_txt

def read_txt(file):
    
    l = []
    with open(cwd + '/' + file + '.txt', "r") as f:
        l = f.read().splitlines()
    
    return l

def preprocess_for_ML(l):

    preprocessed_txt = []
    
    for s in l:
        if s is not np.nan:

            filtered_sentence = nltk.word_tokenize(s.lower())
            filtered_sentence = [w for w in filtered_sentence if not w in stop_words and not w in punctuation] 
            filtered_sentence = ' '.join(filtered_sentence)
            filtered_sentence = [x for x in filtered_sentence if not x in punctuation]
            filtered_sentence = ''.join(filtered_sentence)
            preprocessed_txt.append(filtered_sentence)

        else:

            preprocessed_txt.append(np.nan)     
    
    return preprocessed_txt


def preprocess_for_ML(l):

    preprocessed_txt = []
    
    for s in l:
        if s is not np.nan:

            filtered_sentence = nltk.word_tokenize(s.lower())
            filtered_sentence = [w for w in filtered_sentence if not w in stop_words and not w in punctuation] 
            filtered_sentence = ' '.join(filtered_sentence)
            filtered_sentence = [x for x in filtered_sentence if not x in punctuation]
            filtered_sentence = ''.join(filtered_sentence)
            preprocessed_txt.append(filtered_sentence)

        else:

            preprocessed_txt.append(np.nan)     
    
    return preprocessed_txt

if '__main__' == __name__:

    cwd = os.getcwd()

    STOP_WORDS_NLTK = nltk.corpus.stopwords.words('english')

    stop_words = list(set(list(STOP_WORDS) + list(STOP_WORDS_NLTK)))
    punctuation = list(string.punctuation)

    titles = read_txt('titles')
    titles_annotated = read_txt('titles_annotated')
    abstracts = read_txt('abstracts')
    abstracts_annotated = read_txt('abstracts_annotated')
    has_tf = read_txt('has_tf')
    mammal = read_txt('mammal') 

    titles_preprocessed = preprocess_for_ML(titles)
    titles_annotated_preprocessed = preprocess_for_ML(titles_annotated)
    abstracts_preprocessed = preprocess_for_ML(abstracts)
    abstracts_annotated_preprocessed = preprocess_for_ML(abstracts_annotated)
    has_tf_preprocessed = preprocess_for_ML(has_tf)
    mammal_preprocessed = preprocess_for_ML(mammal) 

    list_as_txt(titles_preprocessed, 'titles_preprocessed.txt')
    list_as_txt(titles_annotated_preprocessed, 'titles_annotated_preprocessed.txt')
    list_as_txt(abstracts_preprocessed, 'abstracts_preprocessed.txt')
    list_as_txt(abstracts_annotated_preprocessed, 'abstracts_annotated_preprocessed.txt')
    list_as_txt(has_tf_preprocessed, 'has_tf_preprocessed.txt')
    list_as_txt(mammal_preprocessed, 'mammal_preprocessed.txt')
