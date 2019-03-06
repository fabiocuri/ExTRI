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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def language_identifier(df):

    ''' Language identifier'''

    lang_abstract = []

    for abstract in df.abstracts:
        lang_abstract.append(detect(abstract))

    df['lang_abstract'] = lang_abstract

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

def categorize_features(l, dictionary):

    l_final = []
    for l_row in l:
        f_ = []
        for f in l_row.split():
            f_.append(dictionary[f])
        l_final.append(f_)

    return l_final

def feature_selection(X, y):

    ''' Feature selection using different algorithms '''

    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(biological_features, labels)
    np.set_printoptions(precision=3)
    print('Chi-squared')
    print(fit.scores_)
    print('')

    model = RandomForestClassifier()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, y)
    print('RFE + LogisticRegression')
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    print('')

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
    # Drop rows without labels
    df = df.loc[df['label'] != ""]
    # Drop NaNs
    df = df.dropna()
    # Drop Non-English abstracts
    df = language_identifier(df)
    df = df.loc[df['lang_abstract'] == 'en']

    d = {}

    for i, column_name in enumerate(df.columns):
        d[column_name] = preprocess(list(df[column_name]))

    # Merge all features

    concat_func_biological_features = lambda a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u: str(
        a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(f) + " " + str(g) + " " + str(
        h) + " " + str(i) + " " + str(j) + " " + str(k) + " " + str(l) + " " + str(m) + " " + str(n) + " " + str(
        o) + " " + str(p) + " " + str(q) + " " + str(r) + " " + str(s) + " " + str(t) + " " + str(u)

    biological_features = list(
        map(concat_func_biological_features, d['Biological process'], d['Cellular component'], d['Chemical compound'],
            d['Disease'], d['Environment'], d['Homo sapiens gene'], d['Molecular function'], d['Organism'],
            d['Phenotype'], d['Tissue'], d['mammal'], d['has_tf'], d['has_exp'], d['chip'], d['cotf'], d['emsa'],
            d['footp'], d['lucifer'], d['y1h'], d['rgene'], d['swblot']))

    # Categorize features and label

    dic = {'false': 0, 'true': 1, 'nan': 2}
    biological_features = categorize_features(biological_features, dic)
    labels = list(d['label'])

    # Feature selection

    feature_selection(biological_features, labels)

    # Final features after feature selection

    concat_func_biological_features = lambda a, b, c, d: str(a) + " " + str(b) + " " + str(c) + " " + str(d)

    biological_features = list(map(concat_func_biological_features, d['has_tf'], d['has_exp'], d['chip'], d['emsa']))

    # Build datasets

    concat_func_text_features = lambda x, y: str(x) + " " + str(y)

    IA = list(map(concat_func_text_features, d['titles'], d['abstracts']))
    IB = list(map(concat_func_text_features, d['titles_annotated'], d['abstracts_annotated']))
    IIA = d['abstracts']
    IIB = d['abstracts_annotated']

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

    # Features
    write_list(biological_features, cwd + '/simulations/biological_features.txt', iterate=True, encoding=None)
