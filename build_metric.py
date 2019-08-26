#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
Date: 26.08.2019
 '''

import pandas as pd
from export_abstracts import read_as_list, write_list
from collections import defaultdict, Counter

def normalize(dictionary, label):

    dic = eval(open("./dictionaries/all_dics.txt2", "r").read())

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

    # Export positive sentences
    df = pd.read_csv('./test/merged_data.csv')
    export, final = [], []
    values = df.values.tolist()

    map_idx = {}
    map_idx['0'] = 'activation'
    map_idx['1'] = 'none'
    map_idx['2'] = 'repression'
    map_idx['3'] = 'undefined'

    for l in values:
        if str(l[3]) != '1':
            export.append(l[7] + '\t' + map_idx[str(l[3])].upper() + '\t' + str(l[5]) + '\t' + str(l[6]) + '\t' + str(l[4]))
        
    export = list(set(export))
    final.append('#PMID:Sentence\tTagRNN\tTF\tTG\tSentence')
    final += export
    write_list(final, './test/predictions.txt', True, 'latin-1')

    silver_standard = read_as_list('ExTRI_confidence', encoding='latin-1')[2:]

    d_silver, d_predicted = defaultdict(list), defaultdict(list)

    for l in silver_standard:
        l_ = l.split('\t')
        if float(l_[3])>float(0.7):
            d_silver[':'.join(l_[0].split(':')[0:2])].append((l_[1], l_[2]))

    for l in final:
        l_ = l.split('\t')
        d_predicted[l_[0]].append((l_[2], l_[3], l_[1]))

    n_silver = normalize(d_silver, False)
    n_predicted = normalize(d_predicted, True)

    export = []

    # Label as TP and FP
    labels = []
    for key in n_predicted.keys():
        for a in n_predicted[key]:
            if (a[0], a[1]) in n_silver[key]:
                export.append(('TP' + '\t' + key + '\t' +  a[0] + '\t' + a[1]+ '\t' + a[2]))
            else:
                export.append(('FP' + '\t' + key + '\t' +  a[0] + '\t' + a[1]+ '\t' + a[2]))

    write_list(export, './coocurrences_positive.txt', True, 'latin-1')
