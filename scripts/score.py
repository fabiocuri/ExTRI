#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao
E-mail: fcuri91@gmail.com
Date: 07.10.2019
'''

import argparse
import pandas as pd
from collections import defaultdict
from export_abstracts import read_as_list, write_list

def normalize(dictionary, label):

    dic = eval(open("../data/dictionaries/all_dics.txt2", "r").read())

    normalized = defaultdict(list)
    missing = []

    for i, l in enumerate(dictionary.keys()):
        for e in dictionary[l]:
            tf, tg = [], []
            word1, word2 = e[0].replace('-', ' '), e[1].replace('-', ' ')

            if word1 in dic.keys():
                tf += [dic[word1]]
            if word1.lower() in dic.keys():
                tf += [dic[word1.lower()]]
            if word2 in dic.keys():
                tg += [dic[word2]]
            if word2.lower() in dic.keys():
                tg += [dic[word2.lower()]]

            if not tf:
                missing.append(word1.lower())

            if not tg:
                missing.append(word2.lower())

            if not tf:
                tf.append('-')
            if not tg:
                tg.append('-')

            for tf_ in list(set(tf)):
                for tg_ in list(set(tg)):
                    if not label:
                        normalized[l].append((tf_, tg_))
                    if label:
                        normalized[l].append((tf_, tg_, e[2]))
        
    return normalized

if '__main__' == __name__:

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--task', type=str, default=None, help="""Triage or relation extraction.""")
    parser.add_argument('--predictions', type=str, default=None, help="""Predictions file.""")

    args = parser.parse_args()

    task = args.task
    predictions = args.predictions

    silver_standard = read_as_list('../data/ExTRI_confidence', encoding='latin-1')[2:]

    positive_abstracts, positive = [], []
    for l in silver_standard:
        l_ = l.split('\t')
        try:
            if l_[11]=='High':
                positive_abstracts.append(str(l_[0].split(':')[0]))
        except:
                continue
    positive_abstracts = list(set(positive_abstracts))

    if task == 'triage':

        labels = read_as_list('../models/' + predictions + '.txt', encoding='latin-1')
        pmids = read_as_list('../triage_list_pmids_test.txt', encoding='latin-1')

        c, d = [], []
        for i, j in zip(labels, pmids):
            if str(i)=='1':
                positive.append(j.split('.')[0])
                if str(j.split('.')[0]) in positive_abstracts:
                    c.append(j)
                if str(j.split('.')[0]) not in positive_abstracts:
                    d.append(j)

        c, d= list(set(c)), list(set(d))
        c_tp, c_fp = len(c), len(d)

        precision = c_tp/(c_tp+c_fp)
        recall = c_tp/(c_tp+(len(positive_abstracts)-c_tp))
        f1 = 2*recall*precision/(recall+precision)

        print(predictions + ': PRECISION = ' + str(precision) + ', RECALL = ' + str(recall) + ', F1-SCORE = ' + str(f1))
        write_list(list(set(positive)), '../triage.output', True, 'latin-1')

    if task == 're':

        labels = read_as_list('../models/' + predictions + '.txt', encoding='latin-1')

        map_idx = {}
        map_idx['0'], map_idx['1'], map_idx['2'], map_idx['3'] = 'activation', 'none', 'repression', 'undefined'
    
        labels = [map_idx[x] for x in labels]
        export, final = [], []
        df = pd.read_csv('../test/re_test.csv')
        values = df.values.tolist()

        for l, label in zip(values, labels):
            if label != 'none':
                positive.append(l[6].split(':')[0])
                export.append(l[6] + '\t' + label.upper() + '\t' + str(l[4]) + '\t' + str(l[5]) + '\t' + str(l[3])[4:])
        
        export = list(set(export))
        final.append('#PMID:Sentence\tTagRNN\tTF\tTG\tSentence')
        final += export

        # Export final positive sentences with their respective TF/TGs
        write_list(final, '../re_sentences.output', True, 'latin-1')

        c, d = [], []
        for i in positive:
                if i in positive_abstracts:
                    c.append(i)
                if i not in positive_abstracts:
                    d.append(i)

        c, d= list(set(c)), list(set(d))
        c_tp, c_fp = len(c), len(d)

        precision = c_tp/(c_tp+c_fp)
        recall = c_tp/(c_tp+(len(positive_abstracts)-c_tp))
        f1 = 2*recall*precision/(recall+precision)

        print(predictions + ': PRECISION = ' + str(precision) + ', RECALL = ' + str(recall) + ', F1-SCORE = ' + str(f1))
        # Export final positive PMIDs
        write_list(list(set(positive)), '../re_pmids.output', True, 'latin-1')

        d_silver, d_predicted = defaultdict(list), defaultdict(list)

        for l in silver_standard:
            l_ = l.split('\t')
            try:
                if l_[11]=='High':
                    d_silver[l_[0].split(':')[0]].append((l_[1], l_[2]))
            except:
                    continue

        for l in final:
            l_ = l.split('\t')
            d_predicted[l_[0].split(':')[0]].append((l_[2], l_[3], l_[1]))

        n_silver = normalize(d_silver, False)
        n_predicted = normalize(d_predicted, True)

        c_tp, c_fp, count = 0, 0, 0
        #Label as TP and FP
        for key in n_predicted.keys():
            for a in n_predicted[key]:
                if (a[0], a[1]) in n_silver[key]:
                    c_tp+=1
                else:
                    c_fp+=1

        for key in n_silver:
            for k in n_silver[key]:
                count+=1

        precision = c_tp/(c_tp+c_fp)
        recall = c_tp/(c_tp+(count-c_tp))
        f1 = 2*recall*precision/(recall+precision)

        print(predictions + ': PRECISION = ' + str(precision) + ', RECALL = ' + str(recall) + ', F1-SCORE = ' + str(f1))
