#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
Date: 26.08.2019
 '''

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from export_abstracts import read_as_list, write_list
from scipy.sparse import hstack
import nltk
import scipy

def run_RF(X, test, labels, data, f):

    np.random.seed(500)

    Encoder = LabelEncoder()
    labels = Encoder.fit_transform(labels)

    if f == 'TF-IDF':
        vect = TfidfVectorizer(max_features=1000)
    if f == 'BoW':
        vect = CountVectorizer(max_features=1000)

    vect.fit(X)
    Train_X_Tfidf = vect.transform(X)
    Test_X_Tfidf = vect.transform(test)

    features_train = build_features(X)
    features_test = build_features(test)

    train = hstack((Train_X_Tfidf, features_train))
    test = hstack((Test_X_Tfidf, features_test))

    RF = RandomForestClassifier(random_state=0, n_estimators=100)
    RF.fit(train,labels)

    y_pred = RF.predict(test)
    write_list(y_pred, cwd + '/models/RF_predictions' + data + '_' + f + '.txt', iterate=True, encoding=encoding)

def build_features(documents):

    list_words = read_as_list('./dictionary_features.txt', encoding='latin-1')
    final = []
   
    for d in documents:
        export = []
        c = defaultdict(lambda: 0)
        for w in nltk.word_tokenize(d):
            c[str(w)] = +1
        for w in list_words:
            if c[str(w)] > 0:
                export += [1]
            else:
                export += [0]

        final.append(export)

    return scipy.sparse.csr_matrix(final)

if '__main__' == __name__:

    ''' Run Random Forest '''

    encoding = 'latin-1'
    cwd = os.getcwd()

    df = pd.read_csv('./train_data.csv')
    labels = list(df['tags'])

    datasets = ['_preprocessed', '_preprocessed_BPE', '_preprocessed_shortest', '_preprocessed_shortest_BPE']
    features = ['TF-IDF', 'BoW']

    for data in datasets:
        for f in features:

            train = read_as_list('./data/train' + data + '.txt', encoding='latin-1')
            test = read_as_list('./data/test' + data + '.txt', encoding='latin-1')

            run_RF(train, test, labels, data, f)
