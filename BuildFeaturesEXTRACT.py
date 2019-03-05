#!/usr/bin/env python
# coding: utf-8

from os import listdir
from os.path import isfile, join
import os
import pandas as pd
from ExportPMIDs import read_as_list

def build_EXTRACT_features(abstracts_PMIDs):
   
   ''' Merge EXTRACT features with handly annotated '''
   
    entities = ['Biological process', 'Cellular component', 'Chemical compound', 'Disease',
                'Environment', 'Homo sapiens gene', 'Molecular function', 'Organism', 'Phenotype', 'Tissue']
      
    annotated_labels = ['mammal', 'has_tf', 'has_exp', 'chip', 'cotf', 'emsa', 'footp', 'lucifer', 'y1h', 'rgene',
                        'swblot', 'label']
   
    abstracts, abstracts_annotated, titles, titles_annotated, pmid_order = [], [], [], [], []

    for i in range(len(entities) + len(annotated_labels)):
        globals()['l_%s' % i] = []

    ann = pd.read_csv(cwd + '/abstracts.all.labeled.csv', sep='\n|\t', encoding=encoding, engine='python')

    for PMID in abstracts_PMIDs:  # Check all PMIDs

        PMID = PMID.split('.')[0]
        journal = read_as_list(cwd + '/abstracts/' + PMID + '.txt', encoding=encoding)
        title = journal[0].split('|')[2]
        abstract = journal[1].split('|')[2]
        titles.append(title)
        abstracts.append(abstract)
        pmid_order.append(PMID)

        for i, label in enumerate(annotated_labels):
            globals()['l_%s' % (i + 10)].append(str(ann.loc[ann['pmid'] == int(PMID)][label].values[0]).upper())

        is_annotated = cwd + '/annotated_abstracts/' + PMID + '_annotated.txt'

        if os.path.isfile(is_annotated):  # If PMID has been annotated...

            annotated_PMID = read_as_list(is_annotated, encoding=encoding)
            l_entities = []
            title_ann = title
            abstract_ann = abstract

            for x in annotated_PMID:

                if x != '':

                    entity = x.split('\t')
                    keyword = entity[0]
                    idx = entities.index(keyword)
                    l_entities.append(idx)
                    words = entity[3]

                    hashtag = '#' + keyword.replace(' ','').upper() + '#'

                    if ';' in words:

                        words = words.split(';')

                        for w in words:
                            title_ann = title_ann.replace(w, hashtag)
                            abstract_ann = abstract_ann.replace(w, hashtag)
                    else:
                        title_ann = title_ann.replace(words, hashtag)
                        abstract_ann = abstract_ann.replace(words, hashtag)

            titles_annotated.append(title_ann)
            abstracts_annotated.append(abstract_ann)
            l_entities, l_non_entities = list(set(l_entities)), list(range(len(entities)))
            not_append = [x for x in l_non_entities if x not in l_entities]

            for e in l_entities:
                globals()['l_%s' % e].append('TRUE')

            for e in not_append:
                globals()['l_%s' % e].append('FALSE')

        else:  # If PMID has not been annotated...

            titles_annotated.append(title)
            abstracts_annotated.append(abstract)

            for e in list(range(len(entities))):
                globals()['l_%s' % e].append('FALSE')

    df = pd.DataFrame()
    df['pmid'] = pmid_order
    df['abstracts'] = abstracts
    df['abstracts_annotated'] = abstracts_annotated
    df['titles'] = titles
    df['titles_annotated'] = titles_annotated

    for i in range(len(entities)):
        df[entities[i]] = globals()['l_%s' % i]
         
    for j in range(len(annotated_labels)):
        idx = j + len(entities)
        df[annotated_labels[j]] = globals()['l_%s' % idx]

    return df

if '__main__' == __name__:

    encoding = "latin-1"
    cwd = os.getcwd()
    abstracts_PMIDs = [f for f in listdir(cwd + '/abstracts') if isfile(join(cwd + '/abstracts', f))]
    data = build_EXTRACT_features(abstracts_PMIDs)
    data.to_csv(cwd + '/data_all_labels.csv', encoding=encoding)
