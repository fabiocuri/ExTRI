#!/usr/bin/env python
# coding: utf-8

''' Author: Fabio Curi Paixao '''

import os
from os import listdir
from selenium import webdriver
import time
from os.path import isfile, join
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from ExportPMIDs import read_as_list, write_list
import pyperclip
from random import shuffle
import argparse
from collections import defaultdict

def find_substring(a_str, sub):

    ''' Find start and end of all matching substrings '''

    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def annotate_with_EXTRACT(abstracts_PMIDs, folder_input, folder_output):
    
    ''' For each abstract in PubTator format, extract entities with EXTRACT and export *.ann file in BRAT format '''

    driver = webdriver.Chrome()
    encoding = "latin-1"

    for PMID in abstracts_PMIDs:

        PMID = PMID.split('.')[0]

        if not os.path.isfile(cwd + folder_output + '/' + PMID + '.ann'): # If PMID has not been yet annotated...

            journal = read_as_list(cwd + folder_input + '/' + PMID + '.txt', encoding=encoding)

            if journal[1].split('|')[2] != '-No abstract-': # If PMID abstract exists...

                # Extract entities from title + abstract

                abstract = journal[1].split('|')[2]
                title = journal[0].split('|')[2]
                merged = title + ' ' + abstract

                driver.get("https://extract.jensenlab.org/")
                wait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Demo']"))).click()
                driver.find_element_by_id("sample_document").clear()
                driver.find_element_by_id('sample_document').send_keys(merged)
                driver.find_element_by_xpath("//input[@type='submit' and @value='Extract Metadata']").click()
                time.sleep(0.1)
                driver.switch_to.frame('result_iframe')
                time.sleep(0.1)

                pyperclip.copy('')

                try: # If any entity is found...

                    driver.find_element_by_id("extract_copy_to_clipboard").click()
                    text = pyperclip.paste()
                    text = text.split('\n')
                    t__, kwds, ents, export = [], [], [], []
                    d, p_idx = defaultdict(list), defaultdict(list)

                    for t in text: # For each entity and keyword found...

                        if t != '':

                            t_ = t.split('\t')
                            kwds.append(t_[3])
                            ents.append(t_[0])

                    # Build dictionary WORD:ENTITY
                    # Unidirectionality: Each word can only have one entity, given that EXTRACT does not export entities by order of appearance, thus ambiguity is not tolerated !
                    # Words with more than one entity are ignored. (reference: Westergaard et al. 2018)

                    for i, j in zip(kwds, ents):

                        if ';' in i:
                            kw = i.split(';')
                            for w in kw:
                                d[w].append(str(j))

                        else:
                            d[i].append(str(j))

                    for word in d:
                        if len(d[word]) == 1: # Strings must have only one entity !
                            idx = list(find_substring(merged, word))
                            for idx_ in idx:
                                p_idx[idx_].append(word)

                    for pos in p_idx:
                        if len(p_idx[pos]) == 1: # String positions must be the start of only one entity !
                            w_ = p_idx[pos][0]
                            l_w_ = len(w_)
                            t__.append((d[w_][0], pos, pos + l_w_, w_))

                    t__ = sorted(t__, key=lambda tup: tup[1])

                    for i, j in enumerate(t__):
                        export.append('T%s' %i + '\t' + str(j[0]) + '\t' + str(j[1]) + '\t' + str(j[2]) + '\t' + j[3])
                        #export.append('#%s' %i + '\t' + str(j[0]) + '\t' + str(j[1]) + '\t' + str(j[2]) + '\t' + j[3])

                    write_list(export, cwd + folder_output + '/' + PMID + '.ann', iterate=True, encoding=encoding)
                    write_list(merged, cwd + folder_output + '/' + PMID + '.txt', iterate=False, encoding=encoding)       
          
                except:
                    print('No entities were found...')

    driver.close()

if '__main__' == __name__:
    
    cwd = os.getcwd()

    parser = argparse.ArgumentParser(description='Hyper-parameters of the model.')
    parser.add_argument('--input', type=str, help="""Folder with PubTator abstracts to be annotated.""")
    parser.add_argument('--output', type=str, help="""Folder with annotated abstracts.""")
    args = parser.parse_args()

    folder_input = '/' + args.input
    folder_output = '/' + args.output

    abstracts_PMIDs = [f for f in listdir(cwd + folder_input) if isfile(join(cwd + folder_input, f))]
    shuffle(abstracts_PMIDs)
    annotate_with_EXTRACT(abstracts_PMIDs, folder_input, folder_output)
