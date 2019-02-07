#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

cwd = os.getcwd()

df_train = pd.DataFrame.from_csv(cwd + '/abstracts.all.labeled.csv', sep='\n|\t', encoding='utf-8')

training_pmid = list(df_train.index)

with open('training_pmid.txt', 'w') as f:
    for item in training_pmid:
        f.write("%s\n" % item)

