#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao
E-mail: fcuri91@gmail.com
Date: 07.10.2019
'''

import argparse
from os import listdir
from export_abstracts import read_as_list, write_list

if '__main__' == __name__:

    ''' Removes wrong genes. '''

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--folder', type=str, help="""Data folder.""")
    args = parser.parse_args()

    to_avoid = read_as_list('../data/to_avoid.txt', encoding='latin-1')
    l_ann = [f for f in listdir('../' + args.folder + '/merged/') if f.endswith('.ann')]

    for a in l_ann:

        ann = read_as_list('../' + args.folder + '/merged/' + a, encoding='latin-1')
        final = []
        for p in ann:
           if p[0] == 'T':
               if p.split('\t')[2].replace(' ','') not in to_avoid:
                   final.append(p)   

        entities = [x.split('\t')[0][1:] for x in final] 
        final+=[x for x in ann if x[0] == '#' and x.split('\t')[0][1:] in entities]

        write_list(list(set(final)), '../' + args.folder + '/merged/' + a, True, 'latin-1')
