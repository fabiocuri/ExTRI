#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
Date: 26.08.2019
 '''

import re
import os
import time
from export_abstracts import read_as_list, write_list
from os import listdir
from flashtext import KeywordProcessor
import argparse

def export_entities(dictionary, list_of_files, label, T, f):

    keyword_processor = KeywordProcessor()

    dic = eval(open("./dictionaries/" + dictionary, "r").read())
    now = time.time()

    for x in dic.keys():
        keyword_processor.add_keyword(x)

    for i, file_name in enumerate(list_of_files):

        file_out = open(f + file_name.replace('txt2', 'out'), 'a')
        sentence = read_as_list(f + file_name, encoding = 'latin-1')

        if sentence and len(sentence) == 1:
            sentence = sentence[0]
            sentence2 = sentence.lower()
            nomes = keyword_processor.extract_keywords(sentence2)
            
            if nomes:
                for nome in nomes:
                    try:
                        p = re.compile(r'\b%s\b' % re.escape(nome))
                        for m in p.finditer(sentence2):
                            file_out.write(str(T)+str(m.start())+'\t' + label + '\t'+str(m.start())+'\t'+str(m.end())+'\t'+str(sentence[m.start():m.end()])+'\t' + '#' + T[1]+str(m.start())+'\t' + 'AnnotatorNotes_' + T+str(m.start())+'\t'+str(dic[nome])+'\n')
                    except:
                        continue

    print('Finished ' + label + ' with time: ' + str(time.time() - now))

if '__main__' == __name__:

    ''' Performs dictionary-mapping NER '''

    cwd = os.getcwd()
    encoding = 'latin-1'

    parser = argparse.ArgumentParser(description='Location of NER files.')
    parser.add_argument('--folder', type=str, help="""Data folder.""")
    args = parser.parse_args()

    list_of_files = [f for f in listdir(args.folder + '/text/') if f.endswith('.txt2')]

    export_entities("dbtf_official_underscore.dic", list_of_files, 'DBTF', 'T1', args.folder + '/text/')
    export_entities("dbtf_syn_underscore.dic", list_of_files, 'DBTF', 'T2', args.folder + '/text/')
    export_entities("dbtf_long_official_underscore.dic", list_of_files, 'DBTF', 'T3', args.folder + '/text/')
    export_entities("dbtf_long_syn_underscore.dic", list_of_files, 'DBTF', 'T4', args.folder + '/text/')
    export_entities("nodbtf_official_underscore.dic", list_of_files, 'NONDBTF', 'T5', args.folder + '/text/')
    export_entities("nodbtf_syn_underscore.dic", list_of_files, 'NONDBTF', 'T6', args.folder + '/text/')
    export_entities("nodbtf_long_official_underscore.dic", list_of_files, 'NONDBTF', 'T7', args.folder + '/text/')
    export_entities("nodbtf_long_syn_underscore.dic", list_of_files, 'NONDBTF', 'T8', args.folder + '/text/')
