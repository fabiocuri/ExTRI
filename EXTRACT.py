#!/usr/bin/env python
# coding: utf-8

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

def annotate_with_EXTRACT(abstracts_PMIDs):

    driver = webdriver.Chrome()
    encoding = "latin-1"

    for PMID in abstracts_PMIDs:

        PMID = PMID.split('.')[0]

        if not os.path.isfile(cwd + '/annotated_abstracts/' + PMID + '_annotated.txt'): # If PMID has not been yet annotated...

            journal = read_as_list(cwd + '/abstracts/' + PMID + '.txt', encoding=encoding)

            if journal[1].split('|')[2] != '-No abstract-': # If PMID abstract exists...

                abstract = journal[1].split('|')[0] + ' ' + journal[1].split('|')[2]

                driver.get("https://extract.jensenlab.org/")
                wait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Demo']"))).click()
                driver.find_element_by_id("sample_document").clear()
                driver.find_element_by_id('sample_document').send_keys(abstract)
                driver.find_element_by_xpath("//input[@type='submit' and @value='Extract Metadata']").click()
                time.sleep(0.1)
                driver.switch_to.frame('result_iframe')
                time.sleep(0.1)

                pyperclip.copy('')

                try:

                    driver.find_element_by_id("extract_copy_to_clipboard").click()
                    text = pyperclip.paste()
                    write_list(text, cwd + '/annotated_abstracts/' + PMID + '_annotated.txt', iterate=False, encoding=encoding)
                    print(str(PMID) + ' : Entities annotated!')

                except:

                    print(str(PMID) + ' : No entities found...')

    driver.close()

if '__main__' == __name__:
    
    cwd = os.getcwd()
    abstracts_PMIDs = [f for f in listdir(cwd + '/abstracts') if isfile(join(cwd + '/abstracts', f))]
    shuffle(abstracts_PMIDs)
    annotate_with_EXTRACT(abstracts_PMIDs)
