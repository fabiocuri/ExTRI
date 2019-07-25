#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
 '''

import os
from export_abstracts import read_as_list, write_list
cwd = os.getcwd()
from os import listdir
import argparse
import nltk
import string
import spacy
import networkx as nx
from networkx import dijkstra_path
import pandas as pd
import time
import traceback

def preprocess_text(l, nlp):

    ''' Preprocess data '''

    cwd = os.getcwd()    

    tokens = nltk.word_tokenize(l.lower())

    for i, t in enumerate(tokens):
        if 'gene1' in t:
            i_gene1 = i
        if 'gene2' in t:
            i_gene2 = i

    WINDOW_SIZE = 5
    # make sure that we don't overflow but using the min and max methods
    FIRST_INDEX = max(i_gene1 - WINDOW_SIZE , 0)
    SECOND_INDEX = min(l.index("gene2") + WINDOW_SIZE, len(tokens))
   
    trimmed_tokens = tokens[FIRST_INDEX : SECOND_INDEX]

    #stop_words = list(set(list(STOP_WORDS) + list(nltk.corpus.stopwords.words('english')) + list(ENGLISH_STOP_WORDS)))
    punctuation = list(string.punctuation)
    #porter = PorterStemmer()

    trimmed_tokens = [w for w in trimmed_tokens if not w in punctuation]
    #trimmed_tokens = [porter.stem(w) for w in trimmed_tokens]
    l = ' '.join([x for x in trimmed_tokens if len(x) > 1 and x != ' ' and x not in punctuation])  # Remove tokens with length of one (noise?)

    document = nlp(l)

    edges, all_edges = [], []
    for token in document:
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_, token.i), '{0}-{1}'.format(child.lower_, child.i)))
            all_edges.append('{0}-{1}'.format(token.lower_, token.i))
            all_edges.append('{0}-{1}'.format(child.lower_, child.i))

    all_edges = list(set(all_edges))
    for i in all_edges:
        if 'gene1' in i:
            n_e1 = i
        if 'gene2' in i:
            n_e2 = i

    graph = nx.Graph(edges)

    try:
        dist = str(nx.shortest_path_length(graph, source=str(n_e1), target=str(n_e2)))
    except:
        dist = 'NONE'

    try:
        path = dijkstra_path(graph, source=str(n_e1), target=str(n_e2), weight='weight')
    except:
        path = 'NONE'

    path = ' '.join([x.split('-')[0] for x in path])

    return l.lower(), path.lower(), dist

def build_data(l_texts, l_ann, type_data):

    nlp = spacy.load('en')

    original, all_sentences, all_sentences_shortest, distances, tags, l_gene1, l_gene2, l_pmids = [], [], [], [], [], [], [], []

    def find_s_e(e, tag):

        word = e.split('\t')[2]
        b = e.split('\t')[1].split(' ')[1]
        e = e.split('\t')[1].split(' ')[2]

        return int(b), int(e), str(word)

    for i, a in enumerate(l_ann):

        if len(a.split('.')[0].split(':')) == 2:
            sentence_index = a.split('.')[0].split(':')[1]
        else:
            sentence_index = 'None'

        print('Remaining: ' + str(len(l_ann)-i))

        already = []
        try:
            ann = read_as_list(cwd + '/' + type_data + '/' + a, encoding=encoding)
            txt = read_as_list(cwd + '/' + type_data + '/' + a.split('.')[0] + '.txt', encoding=encoding)
            txt = ''.join(txt)

            relations = [x for x in ann if x[0] == 'R']

            if type_data == 'train':
                entities = [x for x in ann if x[0] == 'T' and x.split('\t')[1][0:14] != 'AnnotatorNotes']
            else:
                entities = ann

            n_dbtfs = [x for x in entities if x.split('\t')[1].split(' ')[0] == 'DBTF']

            # If there is at least one DBTF and at least two entities...
            if len(n_dbtfs) > 0 and len(entities) > 1:

                # Build positive sentences!
                if relations:
                    for r in relations:

                        tag = r.split('\t')[1].split(' ')[0]
                        ent1, ent2 = r.split('\t')[1].split(' ')[1].split(':')[1], r.split('\t')[1].split(' ')[2].split(':')[1]

                        for e in entities:
                            if e.split('\t')[0] == ent1:
                                b1, e1, word1 = find_s_e(e, ent1)
                            if e.split('\t')[0] == ent2:
                                b2, e2, word2 = find_s_e(e, ent2)

                        invert = False

                        if b1 > b2:
                           invert = True

                        if not invert:
                            out = txt[:b1] + 'gene1' + txt[e1:b2] + 'gene2' + txt[e2:]

                        if invert:
                            out = txt[:b2] + 'gene2' + txt[e2:b1] + 'gene1' + txt[e1:]

                        s_ = nltk.sent_tokenize(out)
                        sentence = []
                        for i, s in enumerate(s_):
                            if "gene1" in s and "gene2" in s:
                                sentence = s
                                idx = i

                        if sentence:
                            for e in entities:
                                sentence = sentence.replace('_', ' ')
                                sentence = sentence.replace('-', ' ')
                                w = e.split('\t')[2].replace('_', ' ')
                                w = ' '.join(w.replace('-', ' ').split())
                                sentence = sentence.replace(w, 'genex')

                            out1, out2, distance = preprocess_text(sentence, nlp)
                            if 'gene1' in out1 and 'gene2' in out1 and 'gene1' in out2 and 'gene2' in out2 and out1 not in all_sentences and out2 not in all_sentences_shortest:
                                all_sentences.append(out1)
                                all_sentences_shortest.append(out2)
                                tags.append(tag.lower())
                                distances.append(distance)
                                if sentence_index == 'None':
                                    export = ' '.join(nltk.sent_tokenize(txt)[int(idx)].replace('_', ' ').replace('-', ' ').split())
                                    original.append(export)
                                    ID = a.split('.')[0] + ':' + str(idx)
                                else:
                                    txt = ' '.join(txt.replace('_', ' ').replace('-', ' ').split())
                                    original.append(txt)
                                    ID = a.split('.')[0]
                                word1, word2 = word1.replace('_', ' '), word2.replace('_', ' ')
                                word1, word2 = ' '.join(word1.replace('-', ' ').split()), ' '.join(word2.replace('-', ' ').split())
                                l_gene1.append(word1)
                                l_gene2.append(word2)
                                l_pmids.append(ID)
                                already.append(word1)
                                already.append(word2)

                # Build negative sentences!
                non_relations = [x for x in entities if ' '.join(x.split('\t')[2].replace('_', ' ').replace('-', ' ').split()) not in already]
                DBTF = [x for x in non_relations if x.split('\t')[1].split(' ')[0] == 'DBTF']
                combinations = [(x,y) for x in DBTF for y in non_relations if y.split('\t')[1].split(' ')[0] == 'DBTF' or y.split('\t')[1].split(' ')[0] == 'NONDBTF']
    
                if combinations:
                    for d in combinations:
                        # Combine all DBTF with every possible other entity different from it. No self-regulation!
                        b1, e1 = int(d[0].split('\t')[1].split(' ')[1]), int(d[0].split('\t')[1].split(' ')[2])
                        word1 = ' '.join(d[0].split('\t')[2].replace('_', ' ').replace('-', ' ').split())                     
                        b2, e2 = int(d[1].split('\t')[1].split(' ')[1]), int(d[1].split('\t')[1].split(' ')[2])
                        word2 = ' '.join(d[1].split('\t')[2].replace('_', ' ').replace('-', ' ').split())  

                        now = time.time()
    
                        if b1 != b2 and word1 != word2:

                            invert = False

                            if b1 > b2:
                                invert = True

                            if not invert:
                                out = txt[:b1] + 'gene1' + txt[e1:b2] + 'gene2' + txt[e2:]
                            if invert:
                                out = txt[:b2] + 'gene2' + txt[e2:b1] + 'gene1' + txt[e1:]

                            s_ = nltk.sent_tokenize(out)
                            sentence = []
                            for i, s in enumerate(s_):
                                if "gene1" in s and "gene2" in s:
                                    sentence = s
                                    idx = i

                            if sentence:
                                for e in entities:
                                    sentence = sentence.replace('_', ' ')
                                    sentence = sentence.replace('-', ' ')
                                    w = e.split('\t')[2].replace('_', ' ')
                                    w = ' '.join(w.replace('-', ' ').split())
                                    sentence = sentence.replace(w, 'genex')

                                out1, out2, distance = preprocess_text(sentence, nlp)
                                if 'gene1' in out1 and 'gene2' in out1 and 'gene1' in out2 and 'gene2' in out2 and out1 not in all_sentences and out2 not in all_sentences_shortest:
                                    all_sentences.append(out1)
                                    all_sentences_shortest.append(out2)
                                    tags.append('none')
                                    distances.append(distance)
                                    if sentence_index == 'None':
                                        export = ' '.join(nltk.sent_tokenize(txt)[int(idx)].replace('_', ' ').replace('-', ' ').split())
                                        original.append(export)
                                        ID = a.split('.')[0] + ':' + str(idx)
                                    else:
                                        txt = ' '.join(txt.replace('_', ' ').replace('-', ' ').split())
                                        original.append(txt)
                                        ID = a.split('.')[0]
                                    l_gene1.append(word1)
                                    l_gene2.append(word2)
                                    l_pmids.append(ID)

        except Exception as e: 
            traceback.print_exc()

    df = pd.DataFrame()
    df['all_sentences'] = all_sentences
    df['all_sentences_shortest'] = all_sentences_shortest
    df['distances'] = distances
    df['tags'] = tags
    df['original'] = original
    df['l_gene1'] = l_gene1
    df['l_gene2'] = l_gene2
    df['l_pmids'] = l_pmids

    df.to_csv(cwd + '/' + type_data + '_data.csv', index = None, header=True)

if '__main__' == __name__:

    ''' Exports data to be used in the models '''

    cwd = os.getcwd()
    encoding = 'latin-1'

    parser = argparse.ArgumentParser(description='Options for relation extraction.')
    parser.add_argument('--folder', type=str, help="""Folder with *.txt and *.ann files.""")
    args = parser.parse_args()

    l_texts = [f for f in listdir(cwd + '/' + args.folder) if f.endswith('.txt')]
    l_ann = [f for f in listdir(cwd + '/' + args.folder) if f.endswith('.ann')]

    build_data(l_texts, l_ann, args.folder)
