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
from sklearn.feature_extraction.text import TfidfVectorizer
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

def run_SVM(X, test, labels):

    np.random.seed(500)

    Encoder = LabelEncoder()
    labels = Encoder.fit_transform(labels)

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(X)
    Train_X_Tfidf = Tfidf_vect.transform(X)
    Test_X_Tfidf = Tfidf_vect.transform(test)

    SVM_linear = svm.SVC(C=1.0, kernel='linear', degree=1, gamma='auto')
    SVM_linear.fit(Train_X_Tfidf,labels)

    y_pred = SVM_linear.predict(Test_X_Tfidf)
    write_list(y_pred, cwd + '/models/predictions.txt', iterate=True, encoding=encoding)

if '__main__' == __name__:

    ''' Run Recurrent Neural Network '''

    encoding = 'latin-1'
    cwd = os.getcwd()

    df = pd.read_csv('./train_data.csv')
    labels = list(df['tags'])

    train = read_as_list('./train_preprocessed.txt', encoding='latin-1')
    test = read_as_list('./test_preprocessed.txt', encoding='latin-1')

    run_SVM(train, test, labels)
