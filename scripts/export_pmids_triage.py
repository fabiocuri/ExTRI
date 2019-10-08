#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao
E-mail: fcuri91@gmail.com
Date: 07.10.2019
'''

import pandas as pd
from export_abstracts import read_as_list, write_list

def remove_duplicates(df, column):

    df = df.drop_duplicates(subset=column, keep="first")
    df = df.dropna()
    df = df.loc[df[column] != ""]

    return df

if '__main__' == __name__:

    ''' Export triage PMIDs and labels of annotated abstracts. '''
    
    encoding = "latin-1"

    ann1 = pd.read_csv('../data/abstracts.all.labeled.csv', sep='\n|\t', encoding=encoding, engine='python')
    ann2 = read_as_list('../data/hackaton_1.tsv', encoding=encoding)
    ann2 = [x.split('\t') for x in ann2]
    ann3 = read_as_list('../data/hackaton_2.tsv', encoding=encoding)
    ann3 = [x.split('\t') for x in ann3]

    di = {True: 1, False: 0}

    ann1['label'].replace(di, inplace=True)

    pmid_1, l_1, pmid_2, l_2, pmid_3, l_3 = list(ann1['pmid']), list(ann1['label']), [x[0] for x in ann2], [x[2] for x in ann2], [x[0] for x in ann3], [x[2] for x in ann3]

    df = pd.DataFrame()
    df['pmid'] = pmid_1 + pmid_2 + pmid_3
    df['label'] = l_1 + l_2 + l_3
    df = df.astype({'pmid': int, 'label': int})
    df = remove_duplicates(df, 'pmid')

    list_PMID = list(df['pmid'])
    list_labels = list(df['label'])
    write_list(list_PMID, '../train_pmids_triage.txt', iterate=True, encoding=encoding)
    write_list(list_labels, '../train_labels_triage.txt', iterate=True, encoding=encoding)
