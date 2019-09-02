#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
Date: 26.08.2019
 '''

import os
import argparse
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from export_abstracts import read_as_list, write_list
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from scipy.sparse import hstack
from RF import build_features

def run_SVM(X, test, labels, data, f):

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

    SVM_linear = svm.SVC(C=1.0, kernel='linear', degree=1, gamma='auto')
    SVM_linear.fit(Train_X_Tfidf,labels)

    y_pred = SVM_linear.predict(Test_X_Tfidf)
    write_list(y_pred, cwd + '/models/SVM_predictions' + data + '_' + f + '.txt', iterate=True, encoding=encoding)

if '__main__' == __name__:

    ''' Run SVM '''

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

            run_SVM(train, test, labels, data, f)
