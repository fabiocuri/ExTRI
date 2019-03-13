#!/usr/bin/env python
# coding: utf-8

''' Author: Fabio Curi Paixao '''

import os
import pandas as pd

def read_as_list(l, encoding):

    ''' Read file as list '''

    l_ = []
    with open(l, "rt", encoding=encoding) as f:
        l_ = f.read().splitlines()
    return l_

def write_list(l, l_name, iterate, encoding):

    ''' Export list '''

    if encoding is not None:
        with open(l_name, 'w', encoding=encoding) as f:
            if iterate:
                for item in l:
                    f.write("%s\n" % item)
            else:
                f.write("%s\n" % l)
    else:
        with open(l_name, 'w') as f:
            if iterate:
                for item in l:
                    f.write("%s\n" % item)
            else:
                f.write("%s\n" % l)

if '__main__' == __name__:
    
    cwd = os.getcwd()
    encoding = "latin-1"

    ann1 = pd.read_csv(cwd + '/data/abstracts.all.labeled.csv', sep='\n|\t', encoding=encoding, engine='python')
    ann2 = read_as_list(cwd + '/data/hackaton_1.tsv', encoding=encoding)
    ann2 = [x.split('\t') for x in ann2]
    ann3 = read_as_list(cwd + '/data/hackaton_2.tsv', encoding=encoding)
    ann3 = [x.split('\t') for x in ann3]

    list_PMID = list(set(list(ann1['pmid']) + [int(x[0]) for x in ann2] + [int(x[0]) for x in ann3]))
    write_list(list_PMID, 'list_PMID.txt', iterate=True, encoding=encoding)
