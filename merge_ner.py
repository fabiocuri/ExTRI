#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
 '''

import os
from export_abstracts import read_as_list, write_list
from os import listdir
import argparse

cwd = os.getcwd()

def merge(f):

    l_ntnu = [f for f in listdir(f + '/NTNU') if f.endswith('.minfner')]
    l_gnormplus = [f for f in listdir(f + '/GNormPlus') if f.endswith('.minfner')]
    l_text = [f for f in listdir(f + '/text') if f.endswith('.txt')]

    for text in l_text:

        rl, write_out, already, final_merge = [], [], [], []

        tx = read_as_list(f + '/text/' + text, encoding=encoding)
        ann = text.split('.')[0] + '.txt.out.minfner'
        ntnu_boolean, gn_boolean = False, False

        if ann in l_ntnu:
            ntnu = read_as_list(f + '/NTNU/' + ann, encoding=encoding)
            ntnu = [' '.join(x.split('\t')[1:]) for x in ntnu if x[0] == 'T'] 
            ntnu = ['N_' + s  for s in ntnu]
            ntnu_boolean = True
        if ann in l_gnormplus:
            gn = read_as_list(f + '/GNormPlus/' + ann, encoding=encoding)
            gn = [' '.join(x.split('\t')[1:]) for x in gn if x[0] == 'T'] 
            gn = ['G_' + s  for s in gn]
            gn_boolean = True

        # Merge both tools and keep only entities
        if ntnu_boolean and gn_boolean:
            entities = ntnu + gn
        elif ntnu_boolean and not gn_boolean:
            entities = ntnu
        elif not ntnu_boolean and gn_boolean:
            entities = gn
        else:
            entities = False

        # Keep all N_DBTF
        final_merge += [x for x in entities if x.split(' ')[0] == 'N_DBTF']
        already = [(x.split(' ')[1], x.split(' ')[2]) for x in final_merge]
        # Keep all G_NONDBTF 
        final_merge += [x for x in entities if x.split(' ')[0] == 'G_NONDBTF' and (x.split(' ')[1], x.split(' ')[2]) not in already]
        already = [(x.split(' ')[1], x.split(' ')[2]) for x in final_merge]
        # Keep all N_NONDBTF
        final_merge += [x for x in entities if x.split(' ')[0] == 'N_NONDBTF' and (x.split(' ')[1], x.split(' ')[2]) not in already]
        already = [(x.split(' ')[1], x.split(' ')[2]) for x in final_merge]
        # Keep all G_DBTF
        final_merge += [x for x in entities if x.split(' ')[0] == 'G_DBTF' and (x.split(' ')[1], x.split(' ')[2]) not in already]

        entities = [x[2:] for x in final_merge]

        if entities:
            i=0
            for e in entities:
                e_ = e.split(' ')
                p, start, end, word = e_[0], e_[1], e_[2], ' '.join(e_[3:])
                write_out.append('T' + str(i) + '\t' + p + ' ' + str(start) + ' ' + str(end) + '\t' + word)
                i += 1

        if write_out:
            write_list(write_out, f + '/merged/' + ann, iterate=True, encoding=encoding)
            write_list(tx, f + '/merged/' + text, iterate=True, encoding=encoding)

if '__main__' == __name__:

    ''' Exports a final NER annotation '''

    encoding = 'latin-1'
    cwd = os.getcwd()

    parser = argparse.ArgumentParser(description='Location of NER files.')
    parser.add_argument('--folder', type=str, help="""Data folder.""")
    args = parser.parse_args()

    merge(cwd + '/' + args.folder)
