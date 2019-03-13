#!/usr/bin/env python
# coding: utf-8

''' Author: Fabio Curi Paixao '''

import os
import pandas as pd
import numpy as np
import nltk
import string
from os import listdir
from os.path import isfile, join
from langdetect import detect
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from ExportPMIDs import read_as_list, write_list
import argparse

def export_training(l, path, iterate, encoding, ext, biological_features=False):

    write_list(l, path + ext, iterate=iterate, encoding=encoding)

    if biological_features:

        l_ = list(map(concat_func_text_features, l, biological_features))
        write_list(l_, path + '_F' + ext, iterate=iterate, encoding=encoding)

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

    ''' Categorize values '''

    l_final = []
    for l_row in l:
        f_ = []
        for f in l_row.split():
            f_.append(dictionary[f])
        l_final.append(f_)

    return l_final

def feature_selection(X, y):

    ''' Feature selection using different algorithms '''

    print('Performing feature selection...')

    test = SelectKBest(score_func=chi2, k=5)
    fit = test.fit(biological_features, labels)
    np.set_printoptions(precision=3)
    print('Chi-squared')
    print(fit.scores_)
    print('')

    model = RandomForestClassifier()
    rfe = RFE(model, 5)
    fit = rfe.fit(X, y)
    print('RFE + LogisticRegression')
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    print('')

def build_EXTRACT_features(abstracts_PMIDs, type_data):

    ''' Merge EXTRACT features with handly annotated '''

    entities = ['Biological process', 'Cellular component', 'Chemical compound', 'Disease',
                'Environment', 'Homo sapiens gene', 'Molecular function', 'Organism', 'Phenotype', 'Tissue']

    abstracts, abstracts_annotated, titles, titles_annotated, pmid_order = [], [], [], [], []

    if type_data == 'train':

        annotated_labels = ['mammal', 'has_tf', 'has_exp', 'chip', 'cotf', 'emsa', 'footp', 'lucifer', 'y1h', 'rgene',
                            'swblot', 'label']

        for i in range(len(entities) + len(annotated_labels)):
            globals()['l_%s' % i] = []

        ann = pd.read_csv(cwd + '/data/abstracts.all.labeled.csv', sep='\n|\t', encoding=encoding, engine='python')

        for PMID in abstracts_PMIDs:  # Check all PMIDs

            PMID = PMID.split('.')[0]
            journal = read_as_list(cwd + '/' + data_folder + '/' + PMID + '.txt', encoding=encoding)
            title = journal[0].split('|')[2]
            abstract = journal[1].split('|')[2]
            titles.append(title)
            abstracts.append(abstract)
            pmid_order.append(PMID)

            for i, label in enumerate(annotated_labels):
                globals()['l_%s' % (i + len(entities))].append(str(ann.loc[ann['pmid'] == int(PMID)][label].values[0]).upper())

            is_annotated = cwd + '/' + entities_folder + '/' + PMID + '.ann'

            if os.path.isfile(is_annotated):  # If PMID has been annotated...

                annotated_PMID = read_as_list(is_annotated, encoding=encoding)
                l_entities = []
                title_ann = title
                abstract_ann = abstract

                for x in annotated_PMID:

                    if x != '':
    
                        entity = x.split('\t')
                        keyword = entity[1]
                        idx = entities.index(keyword)
                        l_entities.append(idx)
                        word = entity[4]

                        hashtag = keyword.replace(' ', '')

                        title_ann = title_ann.replace(word, hashtag)
                        abstract_ann = abstract_ann.replace(word, hashtag)

                titles_annotated.append(title_ann)
                abstracts_annotated.append(abstract_ann)
                l_entities, l_non_entities = list(set(l_entities)), list(range(len(entities)))
                not_append = [x for x in l_non_entities if x not in l_entities]

                for e in l_entities:
                    globals()['l_%s' % e].append('TRUE')
    
                for e in not_append:
                    globals()['l_%s' % e].append('FALSE')

            else:  # If PMID has not been annotated...

                titles_annotated.append(title)
                abstracts_annotated.append(abstract)

                for e in list(range(len(entities))):
                    globals()['l_%s' % e].append('FALSE')

        df = pd.DataFrame()
        df['pmid'] = pmid_order
        df['abstracts'] = abstracts
        df['abstracts_annotated'] = abstracts_annotated
        df['titles'] = titles
        df['titles_annotated'] = titles_annotated

        for i in range(len(entities)):
            df[entities[i]] = globals()['l_%s' % i]

        for j in range(len(annotated_labels)):
            idx = j + len(entities)
            df[annotated_labels[j]] = globals()['l_%s' % idx]

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

    if type_data == 'test':

        for i in range(len(entities)):
            globals()['l_%s' % i] = []

        for PMID in abstracts_PMIDs:  # Check all PMIDs

            PMID = PMID.split('.')[0]
            journal = read_as_list(cwd + '/' + data_folder + '/' + PMID + '.txt', encoding=encoding)
            title = journal[0].split('|')[2]
            abstract = journal[1].split('|')[2]
            titles.append(title)
            abstracts.append(abstract)
            pmid_order.append(PMID)

            is_annotated = cwd + '/' + entities_folder + '/' + PMID + '.ann'

            if os.path.isfile(is_annotated):  # If PMID has been annotated...

                annotated_PMID = read_as_list(is_annotated, encoding=encoding)
                l_entities = []
                title_ann = title
                abstract_ann = abstract

                for x in annotated_PMID:

                    if x != '':
    
                        entity = x.split('\t')
                        keyword = entity[1]
                        idx = entities.index(keyword)
                        l_entities.append(idx)
                        word = entity[4]

                        hashtag = keyword.replace(' ', '')

                        title_ann = title_ann.replace(word, hashtag)
                        abstract_ann = abstract_ann.replace(word, hashtag)

                titles_annotated.append(title_ann)
                abstracts_annotated.append(abstract_ann)
                l_entities, l_non_entities = list(set(l_entities)), list(range(len(entities)))
                not_append = [x for x in l_non_entities if x not in l_entities]

                for e in l_entities:
                    globals()['l_%s' % e].append('TRUE')
    
                for e in not_append:
                    globals()['l_%s' % e].append('FALSE')

            else:  # If PMID has not been annotated...

                titles_annotated.append(title)
                abstracts_annotated.append(abstract)

                for e in list(range(len(entities))):
                    globals()['l_%s' % e].append('FALSE')

        df = pd.DataFrame()
        df['pmid'] = pmid_order
        df['abstracts'] = abstracts
        df['abstracts_annotated'] = abstracts_annotated
        df['titles'] = titles
        df['titles_annotated'] = titles_annotated

        for i in range(len(entities)):
            df[entities[i]] = globals()['l_%s' % i]

        # Drop duplicates
        df = df.drop_duplicates(subset='pmid', keep="last")
        # Drop rows without abstract
        df = df.loc[df['abstracts'] != '-No abstract-']
        # Drop NaNs
        df = df.dropna()
        # Drop Non-English abstracts
        df = language_identifier(df)
        df = df.loc[df['lang_abstract'] == 'en']

    return df

if '__main__' == __name__:

    cwd = os.getcwd()

    parser = argparse.ArgumentParser(description='Hyper-parameters of the model.')
    parser.add_argument('--type', type=str, help="""Train or test data.""")
    parser.add_argument('--data', type=str, help="""Folder with abstracts and titles.""")
    parser.add_argument('--entities', type=str, help="""Folder with entities.""")
    args = parser.parse_args()

    type_data = args.type # Train or test!
    data_folder = args.data # PubTator format
    entities_folder = args.entities # Folder with *.ann

    encoding = "latin-1"
    nltk.download('punkt')
    cwd = os.getcwd()

    stop_words = list(set(list(STOP_WORDS) + list(nltk.corpus.stopwords.words('english'))))
    punctuation = list(string.punctuation)

    abstracts_PMIDs = [f for f in listdir(cwd + '/' + data_folder) if isfile(join(cwd + '/' + data_folder, f))]

    df = build_EXTRACT_features(abstracts_PMIDs, type_data)

    d = {}
    for i, column_name in enumerate(df.columns):
        d[column_name] = preprocess(list(df[column_name]))

    if type_data == 'train':

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

        concat_func_biological_features = lambda a, b, c, d, e, f, g, h: str(a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(f) + " " + str(g) + " " + str(h)

        biological_features = list(map(concat_func_biological_features, d['Homo sapiens gene'], d['has_tf'], d['has_exp'], d['chip'], d['cotf'], d['emsa'], d['lucifer'], d['rgene']))

        concat_func_text_features = lambda x, y: str(x) + " " + str(y)

        # Export training sets !

        export_training(l = list(map(concat_func_text_features, d['titles'], d['abstracts'])), path = cwd + '/simulations/IA', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features)
        export_training(l = list(map(concat_func_text_features, d['titles_annotated'], d['abstracts_annotated'])), path = cwd + '/simulations/IB', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features)
        export_training(l = d['abstracts'], path = cwd + '/simulations/IIA', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features) 
        export_training(l = d['abstracts_annotated'], path = cwd + '/simulations/IIB', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features) 

        export_training(l = d['label'], path = cwd + '/simulations/labels', iterate=True, encoding=None, ext = '.txt', biological_features = False) 

    if type_data == 'test':

        # Final features after feature selection

        concat_func_biological_features = lambda a, b, c, d, e, f, g, h: str(a) + " " + str(b) + " " + str(c) + " " + str(d) + " " + str(e) + " " + str(f) + " " + str(g) + " " + str(h)

        len_ = len(d['Homo sapiens gene'])

        biological_features = list(map(concat_func_biological_features, d['Homo sapiens gene'], [np.nan]*len_, [np.nan]*len_, [np.nan]*len_, [np.nan]*len_, [np.nan]*len_, [np.nan]*len_, [np.nan]*len_))

        concat_func_text_features = lambda x, y: str(x) + " " + str(y)

        # Export test sets !

        export_training(l = list(map(concat_func_text_features, d['titles'], d['abstracts'])), path = cwd + '/simulations/IA_test', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features)
        export_training(l = list(map(concat_func_text_features, d['titles_annotated'], d['abstracts_annotated'])), path = cwd + '/simulations/IB_test', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features)
        export_training(l = d['abstracts'], path = cwd + '/simulations/IIA_test', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features) 
        export_training(l = d['abstracts_annotated'], path = cwd + '/simulations/IIB_test', iterate=True, encoding=None, ext = '.txt', biological_features = biological_features) 
