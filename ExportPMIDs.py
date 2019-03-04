#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

def list_as_txt(l, l_name):

    ''' Export list as txt '''

    with open(l_name, 'w') as f:
        for item in l:
            f.write("%s\n" % item)
            
def read_txt(l):

    ''' Read txt as list '''

    l_ = []
    with open(l, "rt") as f:
        l_ = f.read().splitlines()

    return l_

if '__main__' == __name__:
    
    cwd = os.getcwd()
    
    ann1 = pd.read_csv(cwd + '/abstracts.all.labeled.csv', sep='\n|\t', encoding='utf-8', engine='python')
    ann2 = read_txt(cwd + '/Gold_Label_train.tsv')
    ann3 = read_txt(cwd + '/Gold_Label_test.tsv')
    
    ann2 = [x.split('\t') for x in ann2]
    ann3 = [x.split('\t') for x in ann3]
    
    list_PMID = list(ann1['pmid']) + [int(x[0]) for x in ann2] + [int(x[0]) for x in ann3]
    list_PMID = list(set(list_PMID))
    
    list_as_txt(list_PMID, 'list_PMID.txt')
