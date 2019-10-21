#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao
E-mail: fcuri91@gmail.com
Date: 07.10.2019
'''

import argparse
from os import listdir
from flashtext import KeywordProcessor
from export_abstracts import read_as_list, write_list

def export_entities(keyword_processor, dic, label, T, entities, f, len_title):

    file_out = open(f.replace('txt2', 'out'), 'a')

    for i in entities:
        if i:
            i_ = i.split('\t')
            if int(i_[2]) > int(len_title):
                if keyword_processor.extract_keywords(i_[3].lower()) and i_[3].lower() in dic:
                    file_out.write(str(T)+str(int(i_[1])-int(len_title)-1)+'\t' + label + '\t'+str(int(i_[1])-int(len_title)-1)+'\t'+str(int(i_[2])-int(len_title)-1)+'\t'+str(i_[3])+'\t' + '#' + T[1]+str(int(i_[1])-int(len_title)-1)+'\t' + 'AnnotatorNotes_' + T+str(int(i_[1])-int(len_title)-1)+'\t'+str(dic[i_[3].lower()])+'\n')

def GNormplus_into_BRAT(in_folder, in_folder2):

    ''' Script to convert GNormPlus pubtator into BRAT with DBTF annotation and Entrez normalization.
        in_folder: input folder with .txt sentences annotated by GNormPlus in pubtator format. '''

    files = [f for f in listdir(in_folder) if f.endswith('.txt2')]

    dictionaries = ["dbtf_official_underscore.dic", "dbtf_syn_underscore.dic", "dbtf_long_official_underscore.dic", "dbtf_long_syn_underscore.dic", "nodbtf_official_underscore.dic", "nodbtf_syn_underscore.dic", "nodbtf_long_official_underscore.dic", "nodbtf_long_syn_underscore.dic"]

    for i, dic in enumerate(dictionaries):

        globals()['dic_%s' %i] = eval(open(in_folder2 + "/" + dic, "r").read())
        globals()['keyword_processor_%s' %i] = KeywordProcessor()

        for x in globals()['dic_%s' %i].keys():
            globals()['keyword_processor_%s' %i].add_keyword(x)

    for f in files:

        journal = read_as_list(in_folder + '/' + f, encoding=encoding)
        len_title = len(journal[0].split('|')[2])

        # If entities have been annotated...
        entities = journal[2:-1]

        if entities:
            export_entities(keyword_processor_0, dic_0, 'DBTF', 'T1', entities, in_folder + '/' + f, len_title)
            export_entities(keyword_processor_1, dic_1, 'DBTF', 'T2', entities, in_folder + '/' + f, len_title)
            export_entities(keyword_processor_2, dic_2, 'DBTF', 'T3', entities, in_folder + '/' + f, len_title)
            export_entities(keyword_processor_3, dic_3, 'DBTF', 'T4', entities, in_folder + '/' + f, len_title)
            export_entities(keyword_processor_4, dic_4, 'NONDBTF', 'T5', entities, in_folder + '/' + f, len_title)
            export_entities(keyword_processor_5, dic_5, 'NONDBTF', 'T6', entities, in_folder + '/' + f, len_title)
            export_entities(keyword_processor_6, dic_6, 'NONDBTF', 'T7', entities, in_folder + '/' + f, len_title)
            export_entities(keyword_processor_7, dic_7, 'NONDBTF', 'T8', entities, in_folder + '/' + f, len_title)

if '__main__' == __name__:

    ''' Converts GNormPlus annotation into BRAT. '''

    encoding = 'latin-1'

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--i1', type=str, help="""Data folder.""")
    parser.add_argument('--i2', type=str, help="""Data folder.""")
    args = parser.parse_args()

    GNormplus_into_BRAT(args.i1, args.i2)
