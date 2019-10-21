#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao
E-mail: fcuri91@gmail.com
Date: 07.10.2019
'''

import os
import argparse
import nltk
import spacy
import time
import pandas as pd
from os import listdir
from collections import Counter
from export_abstracts import read_as_list, write_list
from build_data_triage import preprocess, find_experimental_methods

cwd = os.getcwd()

def preprocess_text(l, nlp, all_entities):

    ''' Preprocesses data. '''

    cwd = os.getcwd()    

    tokens = nltk.word_tokenize(l.lower())

    for i, t in enumerate(tokens):
        if 'gene1' in t:
            i_gene1 = i
        if 'gene2' in t:
            i_gene2 = i

    WINDOW_SIZE = 5
    # make sure that we don't overflow but using the min and max methods
    FIRST_INDEX = max(i_gene1 - WINDOW_SIZE , 0)
    SECOND_INDEX = min(l.index("gene2") + WINDOW_SIZE, len(tokens))
   
    trimmed_tokens = tokens[FIRST_INDEX : SECOND_INDEX]

    l = preprocess(' '.join(trimmed_tokens))

    count_entities = Counter(all_entities)

    # Features: DBTF and EXPMETHOD (boolean)
    order_entities = ['EXPMETHOD', 'DBTF']
    for entity in order_entities:
        if entity in count_entities:
            count_entities[entity] = 1
        else:
            count_entities[entity] = 0

        l = str(count_entities[entity]) + ' ' + l

    return l.lower()

def build_data(l_texts, l_ann, type_data, f2, option, out_file):

    ''' Exports data. '''

    nlp = spacy.load('en')

    original, all_sentences, tags, l_gene1, l_gene2, l_pmids = [], [], [], [], [], []

    def find_s_e(e, tag):

        word = e.split('\t')[2]
        b = e.split('\t')[1].split(' ')[1]
        e = e.split('\t')[1].split(' ')[2]

        return int(b), int(e), str(word)

    for i, a in enumerate(l_ann):

        if len(a.split('.')[0].split(':')) == 2:
            sentence_index = a.split('.')[0].split(':')[1]
        else:
            sentence_index = 'None'

        already = []

        try:
            ann = read_as_list(type_data + '/' + a, encoding=encoding)
            txt = read_as_list(type_data + '/' + a.split('.')[0] + '.txt', encoding=encoding)
            txt = ''.join(txt)

            relations = [x for x in ann if x[0] == 'R']

            if type_data == 'train':
                entities = [x for x in ann if x[0] == 'T' and x.split('\t')[1][0:14] != 'AnnotatorNotes']
            else:
                entities = ann

            n_dbtfs = [x for x in entities if x.split('\t')[1].split(' ')[0] == 'DBTF']

            # If there is at least one DBTF and at least two entities...
            if len(n_dbtfs) > 0 and len(entities) > 1:

                # Build positive sentences!
                if relations:
                    for r in relations:

                        tag = r.split('\t')[1].split(' ')[0]
                        ent1, ent2 = r.split('\t')[1].split(' ')[1].split(':')[1], r.split('\t')[1].split(' ')[2].split(':')[1]

                        for e in entities:
                            if e.split('\t')[0] == ent1 and e.split('\t')[1][0:14] != 'AnnotatorNotes':
                                b1, e1, word1 = find_s_e(e, ent1)
                            if e.split('\t')[0] == ent2 and e.split('\t')[1][0:14] != 'AnnotatorNotes':
                                b2, e2, word2 = find_s_e(e, ent2)

                        invert = False

                        if b1 > b2:
                           invert = True

                        if not invert:
                            out = txt[:b1] + 'gene1' + txt[e1:b2] + 'gene2' + txt[e2:]

                        if invert:
                            out = txt[:b2] + 'gene2' + txt[e2:b1] + 'gene1' + txt[e1:]

                        s_ = nltk.sent_tokenize(out)
                        sentence = []
                        for i, s in enumerate(s_):
                            if "gene1" in s and "gene2" in s:
                                sentence = s
                                idx = i

                        if sentence:
                            sentence = sentence.replace('_', ' ')
                            sentence = sentence.replace('-', ' ')
                            for e in entities:
                                if e.split('\t')[0][0] == 'T' and e.split('\t')[1][0:14] != 'AnnotatorNotes':
                                    w = e.split('\t')[2].replace('_', ' ')
                                    w = ' '.join(w.replace('-', ' ').split())
                                    if w:
                                        sentence = sentence.replace(w, 'genex')

                            all_entities = []
                            all_entities.append('DBTF')
                            sentence, all_entities = find_experimental_methods(f2, sentence, all_entities)

                            out1 = preprocess_text(sentence, nlp, all_entities)

                            if 'gene1' in out1 and 'gene2' in out1 and out1 not in all_sentences:
                                all_sentences.append(out1)
                                tags.append(tag.lower())
                                if sentence_index == 'None':
                                    export = ' '.join(nltk.sent_tokenize(txt)[int(idx)].replace('_', ' ').replace('-', ' ').split())
                                    original_ = out1[0:4] + export
                                    original.append(original_)
                                    ID = a.split('.')[0] + ':' + str(idx)
                                else:
                                    txt = ' '.join(txt.replace('_', ' ').replace('-', ' ').split())
                                    original_ = out1[0:4] + txt
                                    original.append(original_)
                                    ID = a.split('.')[0]
                                word1, word2 = word1.replace('_', ' '), word2.replace('_', ' ')
                                word1, word2 = ' '.join(word1.replace('-', ' ').split()), ' '.join(word2.replace('-', ' ').split())
                                l_gene1.append(word1)
                                l_gene2.append(word2)
                                l_pmids.append(ID)
                                already.append(word1)
                                already.append(word2)

                # Build negative sentences!
                non_relations = []
                for e in entities:
                    if e[0] == 'T' and e.split('\t')[1][0:14] != 'AnnotatorNotes':
                        if ' '.join(e.split('\t')[2].replace('_', ' ').replace('-', ' ').split()) not in already:
                            non_relations.append(e)

                DBTF = [x for x in non_relations if x.split('\t')[1].split(' ')[0] == 'DBTF']
                combinations = [(x,y) for x in DBTF for y in non_relations if y.split('\t')[1].split(' ')[0] == 'DBTF' or y.split('\t')[1].split(' ')[0] == 'NONDBTF']
    
                if combinations:
                    for d in combinations:
                        # Combine all DBTF with every possible other entity different from it. No self-regulation!
                        b1, e1 = int(d[0].split('\t')[1].split(' ')[1]), int(d[0].split('\t')[1].split(' ')[2])
                        word1 = ' '.join(d[0].split('\t')[2].replace('_', ' ').replace('-', ' ').split())                     
                        b2, e2 = int(d[1].split('\t')[1].split(' ')[1]), int(d[1].split('\t')[1].split(' ')[2])
                        word2 = ' '.join(d[1].split('\t')[2].replace('_', ' ').replace('-', ' ').split())  

                        now = time.time()
    
                        if b1 != b2 and word1 != word2:

                            invert = False

                            if b1 > b2:
                                invert = True

                            if not invert:
                                out = txt[:b1] + 'gene1' + txt[e1:b2] + 'gene2' + txt[e2:]
                            if invert:
                                out = txt[:b2] + 'gene2' + txt[e2:b1] + 'gene1' + txt[e1:]

                            s_ = nltk.sent_tokenize(out)
                            sentence = []
                            for i, s in enumerate(s_):
                                if "gene1" in s and "gene2" in s:
                                    sentence = s
                                    idx = i

                            if sentence:
                                sentence = sentence.replace('_', ' ')
                                sentence = sentence.replace('-', ' ')
                                for e in entities:
                                    if e.split('\t')[0][0] == 'T' and e.split('\t')[1][0:14] != 'AnnotatorNotes':
                                        w = e.split('\t')[2].replace('_', ' ')
                                        w = ' '.join(w.replace('-', ' ').split())
                                        sentence = sentence.replace(w, 'genex')

                                all_entities = []
                                all_entities.append('DBTF')
                                sentence, all_entities = find_experimental_methods(f2, sentence, all_entities)
                                out1 = preprocess_text(sentence, nlp, all_entities)

                                if 'gene1' in out1 and 'gene2' in out1 and out1 not in all_sentences:
                                    all_sentences.append(out1)
                                    tags.append('none')
                                    if sentence_index == 'None':
                                        export = ' '.join(nltk.sent_tokenize(txt)[int(idx)].replace('_', ' ').replace('-', ' ').split())
                                        original_ = out1[0:4] + export
                                        original.append(original_)
                                        ID = a.split('.')[0] + ':' + str(idx)
                                    else:
                                        txt = ' '.join(txt.replace('_', ' ').replace('-', ' ').split())
                                        original_ = out1[0:4] + txt
                                        original.append(original_)
                                        ID = a.split('.')[0]
                                    l_gene1.append(word1)
                                    l_gene2.append(word2)
                                    l_pmids.append(ID)

        except Exception as e: 
            continue

    df = pd.DataFrame()
    df['all_sentences'] = all_sentences
    df['tags'] = tags
    df['original'] = original
    df['l_gene1'] = l_gene1
    df['l_gene2'] = l_gene2
    df['l_pmids'] = l_pmids

    if option == 'test':
        df.to_csv(out_file + '/re_test.csv', index=False)

    write_list(tags, out_file + '/re_'  + option + '_labels.txt', iterate=True, encoding=encoding)
    write_list(original, out_file + '/re_' + option + '_original.txt', iterate=True, encoding=encoding)
    write_list(all_sentences, out_file + '/re_' + option + '_preprocessed.txt', iterate=True, encoding=encoding)

if '__main__' == __name__:

    ''' Exports relation extraction data. '''

    encoding = 'latin-1'

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--i1', type=str, help="""Folder with abstracts and annotations.""")
    parser.add_argument('--i2', type=str, help="""Data folder.""")
    parser.add_argument('--option', type=str, help="""Train or test.""")
    parser.add_argument('--o', type=str, help="""Output folder.""")
    args = parser.parse_args()

    l_texts = [f for f in listdir(args.i1) if f.endswith('.txt')]
    l_ann = [f for f in listdir(args.i1) if f.endswith('.ann')]

    build_data(l_texts, l_ann, args.i1, args.i2, args.option, args.o)
