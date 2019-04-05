#!/usr/bin/env python
# coding: utf-8

import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, '../Triage/')
from ExportPMIDs import read_as_list, write_list
import pandas as pd
from collections import Counter
import pickle
import string

def sanity_check(l1, l2):
    
    for l in l1:
        l1_ = read_as_list(cwd + '/NER-merged/train_abstracts/' +  l, encoding = encoding)
        l2_ = read_as_list('../data/hackaton/' +  l, encoding = encoding)
        
        l1_ = ' '.join(l1_)
        l2_ = ' '.join(l2_)
        
        assert len(l1_) == len(l2_), 'Oh no!!!'

def load_obj(name):
    
    with open(name + '.pkl', 'rb') as f:
        
        return pickle.load(f)

def strip_punctuation(s):

    punctuation = list(string.punctuation)
    
    for p in punctuation:
        s = s.replace(p, '')
        
    return s

def list_TFs(TFdic):

    l = []

    for x in TFdic:

        x_ = strip_punctuation(x.lower())
        l_ = [strip_punctuation(k.lower()) for k in TFdic[x]]
        l += [x_] + l_

    l = list(set(l))

    return l

def tag_DBTF(name_, tag_, l_TF):
    
    ''' Detect DbTFs and add gene normalization line from Entrez '''

    word_ = strip_punctuation(name_.lower())
                    
    # Check whether genes are DbTF !
    
    possibilities = ['ABBREVIATION', 'FULL_NAME', 'NO_CLASS', 'NESTED']
    
    if tag_ in possibilities:

        if word_ in l_TF:

            tag_ = 'DBTF'

        else: 

            tag_ = 'NONDBTF'  

    return tag_

def merge_tools(l1, l2, l_TF):

    for PMID in l1:
    
        l1_ = read_as_list('../data/hackaton_ann/' +  PMID, encoding = encoding)
        l2_ = read_as_list(cwd + '/NER-merged/train_abstracts/' +  PMID, encoding = encoding)
        
        idx, tag, start, end, name = [], [], [], [], []
        
        l1_ = [x.split('\t') for x in l1_]
        l2_ = [x.split('\t') for x in l2_]
        
        r = {}
        
        for el in l1_:
            
            break_ = el[1].split(' ')
            tag.append(break_[0])
            
            if el[0][0] == 'T':
                
                idx.append(el[0] + '_HACK')
                start.append(int(break_[1]))
                end.append(break_[2])
                name.append(el[2])
                
            if el[0][0] == '#':
                
                idx.append(break_[1] + '_HACK')
                start.append('')
                end.append('')
                name.append(el[2])
                
            if el[0][0] == 'R':
                
                where = idx.index(break_[1].split(':')[1] + '_HACK')
                r[break_[1].split(':')[1]] = start[where]
                
                where = idx.index(break_[2].split(':')[1] + '_HACK')
                r[break_[2].split(':')[1]] = start[where]
                
                idx.append(el[0])
                start.append(break_[1])
                end.append(break_[2])
                name.append('')
                
        for el in l2_:
            
            break_ = el[1].split(' ')
            tag.append(break_[0])
            
            if el[0][0] == 'T':
                
                idx.append(el[0] + '_NEW')
                start.append(int(break_[1]))
                end.append(break_[2])
                name.append(el[2])
                
            if el[0][0] == '#':
                
                idx.append(break_[1] + '_NEW')
                start.append('')
                end.append('')
                name.append(el[2])
                
        df = pd.DataFrame()
        
        df['idx'] = idx
        df['tag'] = tag
        df['start'] = start
        df['end'] = end
        df['name'] = name
        
        # Merge both datasets, keeping the hackaton if not Gene.
        
        df['start_type'] = [str(type(x)) for x in list(df['start'])]

        df_start_int = df[df['start_type'] == "<class 'int'>"]
        df_hashtag_R = df[df['start_type'] != "<class 'int'>"]
        
        df_start_int = df_start_int.sort_values(by = ['start', 'idx'], ascending=[True, True])
        
        counter_start = Counter(list(df_start_int['start']))
        
        list_doubles, list_good = [], []
        
        for key in counter_start:
            if counter_start[key] > 1:
                list_doubles.append(key)
            else:
                list_good.append(key)

        df_good = df_start_int[df_start_int['start'].isin(list_good)]
        
        important_list = ['ABBREVIATION', 'FULL_NAME', 'NO_CLASS', 'NESTED', 'DBTF', 'NONDBTF']
        priority_list = ['ABBREVIATION', 'FULL_NAME', 'NO_CLASS', 'NESTED']
        low_priority = ['SPECIES', 'CELLULARCOMPONENT', 'BIOLOGICALPROCESS', 'DISEASE', 'ORGANISM', 'MOLECULARFUNCTION', 'CHEMICALCOMPOUND', 'TISSUE']

        for d in list_doubles:

            df_ = df_start_int[df_start_int['start'] == d]
            d_backup = df_
            df_ = df_[df_['tag'].isin(important_list)]

            if df_.empty:
                df_ = d_backup[d_backup['tag'].isin(low_priority)]
                
            if df_.empty:
                print('none of the priorities....')
                print(d_backup)

            if len(df_) > 1:
                df_ = df_[df_['tag'].isin(priority_list)]

            df_good = pd.concat([df_good, df_])
            
        for key, st in zip(list(df_good['idx']), list(df_good['start'])):

            df_ = df_hashtag_R[df_hashtag_R['idx'] == key]

            try: # if annotation exists

                df_['start'][df_.index[0]] = st
                df_good = pd.concat([df_good, df_])

            except: 
                continue
            
        df_good = df_good.sort_values(by = ['start', 'idx'], ascending=[True, True])
        df_good.index = list(range(len(df_good)))
        
        t = []
        i, j = 1, 1
        
        for x in list(df_good.index):
            
            if x % 2 == 0:
                t.append('T%s' %i)
                i+=1
                
            else:
                t.append('#%s' %j)
                j+=1

        df_good['idx'] = t
        
        df_relations = df_hashtag_R[df_hashtag_R['start'] != '']

        if not df_relations.empty:
        
            df = pd.DataFrame()
        
            l1, l2, l3, l4 = [], [], [], []
            
            p = 1
        
            for idx_, tag_, start_, end_ in zip(df_relations['idx'], df_relations['tag'], df_relations['start'], df_relations['end']):
                
                l1.append('R' + str(p))
                p+=1
                l2.append(tag_)
                
                fragment1 = df_good[df_good['start'] == r[start_.split(':')[1]]]
                fragment2 = df_good[df_good['start'] == r[end_.split(':')[1]]]
                
                for i, j in zip(list(fragment1['idx']), list(fragment2['idx'])):
                    
                    if i[0] == 'T':
                        
                        l3.append('Arg1:' + i)
                        
                    if j[0] == 'T':
                        
                        l4.append('Arg2:' + j)

            df['idx'] = l1
            df['tag'] = l2
            df['start'] = l3
            df['end'] = l4
            df['name'] = [' '] * len(l3)
            df['start_type'] = [' '] * len(l3)
            
            df_ = pd.concat([df_good, df])
            
        else:
            
            df_ = df_good
            
        # Delete FAMILY, SEQUENCE and MULTIPLE
    
        idxs_to_delete, rel_remove1, rel_remove2 = [], [], []

        for idx_, tag_ in zip(list(df_['idx']), list(df_['tag'])):

            if tag_ == 'FAMILY' or tag_ == 'SEQUENCE' or tag_ == 'MULTIPLE':

                idxs_to_delete.append(str(idx_))
                idxs_to_delete.append('#' + str(idx_[1:]))
                rel_remove1.append('Arg1:' + str(idx_))
                rel_remove2.append('Arg2:' + str(idx_))
                
        df_ = df_[~df_['idx'].isin(idxs_to_delete)]
        df_ = df_[~df_['start'].isin(rel_remove1)]
        df_ = df_[~df_['end'].isin(rel_remove2)]
        
        # Delete NO_CLASS without Entrez ID

        idxs_to_delete, rel_remove1, rel_remove2 = [], [], []

        for idx_, idx__, tag_ in zip(list(df_.index), list(df_['idx']), list(df_['tag'])):

            if tag_ == 'NO_CLASS':

                try:
                    
                    out = int(df_['name'][idx_+1])

                except:

                    idxs_to_delete.append(idx__)
                    idxs_to_delete.append('#' + idx__[1:])
                    rel_remove1.append('Arg1:' + str(idx__))
                    rel_remove2.append('Arg2:' + str(idx__))

        df_ = df_[~df_['idx'].isin(idxs_to_delete)]
        df_ = df_[~df_['start'].isin(rel_remove1)]
        df_ = df_[~df_['end'].isin(rel_remove2)]
        
        df_.index = list(range(len(df_)))
        
        # Export in BRAT format !
        
        export = []
        
        for idx_, tag_, start_, end_, name_ in zip(df_['idx'], df_['tag'], df_['start'], df_['end'], df_['name']):
            
            if idx_[0] == 'T':
                
                tag = tag_DBTF(name_, tag_, l_TF)
            
                export.append(str(idx_) + '\t' + str(tag) + ' ' + str(start_) + ' ' + str(end_) + '\t' + str(name_))
                
            if idx_[0] == '#':
                
                export.append(str(idx_) + '\t' + str(tag_) + ' T' + str(idx_.replace('#', '')) + '\t' + str(name_))
                             
            if idx_[0] == 'R':
                
                export.append(str(idx_) + '\t' + str(tag_) + ' ' + str(start_) + ' ' + str(end_))
                
        write_list(export, cwd + '/final_training/' + PMID.split('.')[0] + '.ann', iterate=True, encoding=encoding)    

if '__main__' == __name__:

    cwd = os.getcwd()
    
    encoding = 'latin-1'
    
    new_articles = [f for f in listdir(cwd + '/NER-merged/train_abstracts') if f.endswith('.txt')]
    hackaton_articles = [f for f in listdir('../data/hackaton') if f.endswith('.txt')]
    sanity_check(hackaton_articles, new_articles)

    new_annotations = [f for f in listdir(cwd + '/NER-merged/train_abstracts') if f.endswith('.ann')]
    hackaton_annotations = [f for f in listdir('../data/hackaton_ann')]

    TFdic = load_obj('../data/TFdic')
    l_TF = list_TFs(TFdic)

    merge_tools(hackaton_annotations, new_annotations, l_TF)
