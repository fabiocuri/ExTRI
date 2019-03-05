#!/usr/bin/env python
# coding: utf-8

import string
import os
import pandas as pd
import nltk
from langdetect import detect
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS
from ExportPMIDs import write_list

def translate(l):
    l = TextBlob(l)
    try:
        l = str(l.translate(to="en"))
    except:
        l = l

    return l

def dataframe_in_english(df):
    ''' Make sure all data is in English '''

    # Language identification to make sure all papers are in English

    lang_abstract = []

    for abstract in df.abstracts:
        lang_abstract.append(detect(abstract))

    df['lang_abstract'] = lang_abstract
    list_languages = list(set(lang_abstract))
    foreign_languages = [x for x in list_languages if x not in ['en']]

    for FL in foreign_languages:

        new_df = df.loc[df['lang_abstract'] == FL]

        for abstract, abstract_annotated, title, title_annotated, PMID in zip(new_df['abstracts'],
                                                                              new_df['abstracts_annotated'],
                                                                              new_df['titles'],
                                                                              new_df['titles_annotated'],
                                                                              new_df['pmid']):
            df.abstracts[PMID] = translate(abstract)
            df.abstracts_annotated[PMID] = translate(abstract_annotated)

            df.titles[PMID] = translate(title)
            df.titles_annotated[PMID] = translate(title_annotated)

    return df

def preprocess(l):
    ''' Preprocess data '''

    l = [str(x) for x in l]

    preprocessed_l = []

    for s in l:
        filtered_sentence = nltk.word_tokenize(s.lower())
        filtered_sentence = [w for w in filtered_sentence if not w in stop_words and not w in punctuation]
        filtered_sentence = ' '.join(filtered_sentence)
        filtered_sentence = [x for x in filtered_sentence if not x in punctuation]
        filtered_sentence = ''.join(filtered_sentence)
        preprocessed_l.append(filtered_sentence)

    return preprocessed_l


if '__main__' == __name__:

    nltk.download('punkt')
    cwd = os.getcwd()

    stop_words = list(set(list(STOP_WORDS) + list(nltk.corpus.stopwords.words('english'))))
    punctuation = list(string.punctuation)

    df = pd.read_csv(cwd + '/data_all_labels.csv', engine='python')
    df = df.drop(columns=['Unnamed: 0'])

    # Drop duplicates
    df = df.drop_duplicates(subset='pmid', keep="last")

    # Drop rows without abstract
    df = df.loc[df['abstracts'] != '-No abstract-']

    df = dataframe_in_english(df)

    d = {}

    for i, column_name in enumerate(df.columns):
        d[column_name] = preprocess(list(df[column_name]))

    # Build datasets

    concat_func_text_features = lambda x, y: str(x) + " " + str(y)

    IA = list(map(concat_func_text_features, d['titles'], d['abstracts']))
    IB = list(map(concat_func_text_features, d['titles_annotated'], d['abstracts_annotated']))
    IIA = d['abstracts']
    IIB = d['abstracts_annotated']

    concat_func_biological_features = lambda a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u: str(
        a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(f) + " " + str(g) + " " + str(
        h) + " " + str(i) + " " + str(j) + " " + str(k) + " " + str(l) + " " + str(m) + " " + str(n) + " " + str(
        o) + " " + str(p) + " " + str(q) + " " + str(r) + " " + str(s) + " " + str(t) + " " + str(u)

    biological_features = list(
        map(concat_func_biological_features, d['Biological process'], d['Cellular component'], d['Chemical compound'],
            d['Disease'], d['Environment'], d['Homo sapiens gene'], d['Molecular function'], d['Organism'],
            d['Phenotype'], d['Tissue'], d['mammal'], d['has_tf'], d['has_exp'], d['chip'], d['cotf'], d['emsa'],
            d['footp'], d['lucifer'], d['y1h'], d['rgene'], d['swblot']))

    IA_F = list(map(concat_func_text_features, IA, biological_features))
    IB_F = list(map(concat_func_text_features, IB, biological_features))
    IIA_F = list(map(concat_func_text_features, IIA, biological_features))
    IIB_F = list(map(concat_func_text_features, IIB, biological_features))

    # X training
    write_list(IA, cwd + '/simulations/IA.txt', iterate=True, encoding=None)
    write_list(IB, cwd + '/simulations/IB.txt', iterate=True, encoding=None)
    write_list(IIA, cwd + '/simulations/IIA.txt', iterate=True, encoding=None)
    write_list(IIB, cwd + '/simulations/IIB.txt', iterate=True, encoding=None)

    # X training with features
    write_list(IA_F, cwd + '/simulations/IA_F.txt', iterate=True, encoding=None)
    write_list(IB_F, cwd + '/simulations/IB_F.txt', iterate=True, encoding=None)
    write_list(IIA_F, cwd + '/simulations/IIA_F.txt', iterate=True, encoding=None)
    write_list(IIB_F, cwd + '/simulations/IIB_F.txt', iterate=True, encoding=None)

    # y training set
    write_list(d['label'], cwd + '/simulations/labels.txt', iterate=True, encoding=None)
