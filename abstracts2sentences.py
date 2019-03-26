#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from ExportPMIDs import read_as_list, write_list
from nltk import tokenize

if '__main__' == __name__:

    cwd = os.getcwd()

    encoding = 'latin-1'

    parser = argparse.ArgumentParser(description='Convert abstracts into sentences.')
    parser.add_argument('--abstracts', type=str, help="""Train or test data.""")
    args = parser.parse_args()

    abstracts_folder = cwd + '/pubtator/' + args.abstracts
    output_folder = cwd + '/pubtator/' + args.abstracts.split('_')[0] + '_sentences'
    
    abstracts_PMIDs = [f for f in os.listdir(abstracts_folder) if os.path.isfile(os.path.join(abstracts_folder, f))]

    for PMID in abstracts_PMIDs:
    
        PMID_ = PMID.replace('.txt', '')
        journal = read_as_list(abstracts_folder + '/' + PMID, encoding=encoding) # Read journals in PubTator format
        title = journal[0].split('|')[2]
        abstract = journal[1].split('|')[2]
        a_ = tokenize.sent_tokenize(abstract)
    
        if a_[0] != '-No abstract-':
            for i, s in enumerate(a_):
                l_ = []
                l1 = PMID_ + '|t|' + title
                l2 = PMID_ + '|a|' + s
                l_.append(l1)
                l_.append(l2)
                l_.append('')
                write_list(l_, output_folder + '/' + PMID_ + '_' + str(i) + '.txt', iterate=True, encoding=encoding)
