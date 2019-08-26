#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
Date: 26.08.2019
 '''

import os
import re
import argparse

def read_as_list(l, encoding):

    ''' Read file as list '''

    l_ = []
    with open(l, "rt", encoding=encoding) as f:
        l_ = f.read().splitlines()
    return l_

def write_list(l, l_name, iterate, encoding):

    ''' Export list '''

    with open(l_name, 'w', encoding=encoding) as f:
        if iterate:
            for item in l:
                f.write("%s\n" % item)
        else:
            f.write("%s\n" % l)

if '__main__' == __name__:

    ''' Exports pubtator abstracts as .txt '''

    cwd = os.getcwd()
    encoding = 'latin-1'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder', type=str, help="""Data folder.""")
    args = parser.parse_args()

    in_folder = cwd + '/' + args.folder + '/pubtator/'
    out_folder = cwd + '/' + args.folder + '/text/'
    articles = [f for f in os.listdir(in_folder) if f.endswith('.txt')]

    for PMID in articles:
        article = read_as_list(in_folder + PMID, encoding=encoding)
        try:
            article = article[-2].split('|')[2]
            article = re.sub(r'[^\x00-\x7f]', r' ', article)  # ASCII
            write_list(article, out_folder + PMID, iterate=False, encoding=encoding)
        except:
            continue
