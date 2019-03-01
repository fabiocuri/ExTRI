#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

def list_as_txt(l, l_name):

    ''' Export list as txt '''

    with open(l_name, 'w') as f:
        for item in l:
            f.write("%s\n" % item)

if '__main__' == __name__:
    
    cwd = os.getcwd()
    df = pd.read_csv(cwd + '/abstracts.all.labeled.csv', sep='\n|\t', encoding='utf-8', engine='python')
    training_pmid = list(df.pmid)

    list_as_txt(training_pmid, 'training_pmid.txt')

