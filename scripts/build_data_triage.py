#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao
E-mail: fcuri91@gmail.com
Date: 07.10.2019
'''

import nltk
import string
import argparse
from os import listdir
from collections import Counter
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from export_abstracts import read_as_list, write_list
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

def preprocess(l):

    ''' Preprocesses data. '''

    stop_words = list(set(list(STOP_WORDS) + list(nltk.corpus.stopwords.words('english')) + list(ENGLISH_STOP_WORDS)))
    punctuation = list(string.punctuation)
    porter = PorterStemmer()

    l = nltk.word_tokenize(str(l).lower())
    l = [w for w in l if not w in stop_words and not w in punctuation]
    l = ' '.join([porter.stem(w) for w in l])
    l = ''.join([x for x in l if x not in punctuation])
    l = ' '.join([x for x in l.split(' ') if len(x) > 1 and x != ' ']) # Remove tokens with length of one (noise?)

    return l

def find_experimental_methods(f, txt, all_entities):

    ''' Finds experimental methods. '''

    encoding = 'latin-1'
    exp_methods = read_as_list(f + '/experimental_methods.txt', encoding=encoding)

    for e in exp_methods:
        if ' ' + e + ' ' in txt:
            n = txt.count(e)
            txt = txt.replace(e, 'EXPMETHOD')
            for i in range(n):
                all_entities.append('EXPMETHOD')

    return txt, all_entities

def build_data(folder, folder2, folder3, pubtator_articles, option):

    ''' Exports data. '''

    out, out_original, labels, list_pmids = [], [], [], []

    for txt in pubtator_articles:

        text_original = read_as_list(folder +  '/' + txt, encoding=encoding)[0]
        text = text_original
        PMID = txt.split('.')[0]
        ann = read_as_list(folder + '/' + PMID + '.ann', encoding=encoding)  # Read annotations
        ann = [x for x in ann if x[0] == 'T']

        if option == 'train':

            label = read_as_list(folder2 + '/triage_train_labels.txt', encoding=encoding)  # Read annotations
            PMIDs = read_as_list(folder2 + '/triage_train_pmids.txt', encoding=encoding)  # Read annotations
            label_txt = label[PMIDs.index(PMID)]

        lookback, delete, all_entities = [], [], []

        # Build list of start and end of sentences and delete entities inside of entities ...
        for x in ann:
            lookback.append((x.split('\t')[1].split(' ')[1] + ' ' + x.split('\t')[1].split(' ')[2]))

        for x in lookback:
            copy = lookback.copy()
            copy.remove(x)
            start = x.split(' ')[0]

            for el in copy:
                if int(start) in range(int(el.split(' ')[0]), int(el.split(' ')[1])):
                    delete.append(start)

        ann = [x.split('\t') for x in ann if x.split('\t')[1].split(' ')[1] not in delete]
        ann = [x[1].split(' ') for x in ann]
        ann.sort(key=lambda x: int(x[1]))

        c = 0
        for a in ann:
            tag, start, end = a[0], int(a[1]) + c, int(a[2]) + c
            all_entities.append(tag)
            c+= len(tag) - end + start
            text = text[:start] + tag + text[end:]

        # Find experimental settings !
        text, all_entities = find_experimental_methods(folder3, text, all_entities)

        text = preprocess(text)
        
        count_entities = Counter(all_entities)

        # Features: DBTF and EXPMETHOD (boolean)
        order_entities = ['EXPMETHOD', 'DBTF']
        for entity in order_entities:

            if entity in count_entities:
                count_entities[entity] = 1
            else:
                count_entities[entity] = 0

            text = str(count_entities[entity]) + ' ' + text
            text_original = str(count_entities[entity]) + ' ' + text_original

        out.append(text)
        out_original.append(text_original)

        if option == 'train':
            labels.append(label_txt)
        if option == 'test':
            list_pmids.append(':'.join(txt.split(':')[0:2]))

    return out, out_original, labels, list_pmids

if '__main__' == __name__:

    ''' Exports triage data. '''

    encoding = "latin-1"
    nltk.download('punkt')
    nltk.download('stopwords')

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--i1', type=str, help="""Folder with abstracts and annotations.""")
    parser.add_argument('--i2', type=str, help="""Data folder.""")
    parser.add_argument('--i3', type=str, help="""Data folder.""")
    parser.add_argument('--option', type=str, help="""Train or test.""")
    parser.add_argument('--o', type=str, help="""Output folder.""")
    args = parser.parse_args()

    pubtator_articles = [f for f in listdir(args.i1) if f.endswith('.txt')]

    # Replace words by entities!
    data, data_original, labels, list_pmids = build_data(args.i1, args.i2, args.i3, pubtator_articles, args.option)

    if args.option == 'test':
        write_list(list_pmids, args.o + '/triage_test_pmids.txt', iterate=True, encoding=encoding)  

    write_list(data, args.o + '/triage_' + args.option + '_preprocessed.txt', iterate=True, encoding=encoding)  
    write_list(data_original, args.o + '/triage_' + args.option + '_original.txt', iterate=True, encoding=encoding)  
    write_list(labels, args.o + '/triage_' + args.option + '_labels.txt', iterate=True, encoding=encoding)  
