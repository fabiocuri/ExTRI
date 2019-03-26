Author: Fabio Curi Paixao

Encoding: latin-1

## Data: 

data/abstracts.all.labeled.csv

data/hackaton_1.tsv

data/hackaton_2.tsv

## Installations:

1. pip3 install -r requirements

2. Download GNormPlus (both Perl and Java's version) and install them correctly: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/

3. Download GloVe and install it correctly: https://nlp.stanford.edu/projects/glove/

# Get articles and sentences

1. Obtain list of PMIDs 

   * python3 ExportPMIDs.py

Note that if you already have a list of PMIDs, you can name your files 'PMID_train.txt' and 'PMID_test.txt' and move to the next step.

2. Retrieve articles in PubTator format 

   * Copy retrieve_articles_triage.sh into the GNormPlusPerl folder and do bash retrieve_articles_triage.sh

3. Export sentences in PubTator format

   python3 abstracts2sentences.py --abstracts {train_abstracts, test_abstracts}

# NER

## GNormPlus

   * Copy GNormPlus.sh into the GNormPlusJava folder and do ´bash GNormPlus.sh´

## EXTRACT (https://extract.jensenlab.org/)

   * python3 EXTRACT.py --input {train_abstracts, train_sentences, test_abstracts, test_sentences}

## Build dictionaries of DbTF

   * Build DbTF dictionary through python3 BuildTFdic.py

## Merge GNormPlus and EXTRACT

   * python3 MergeNER.py --input {train_abstracts, train_sentences, test_abstracts, test_sentences}

## Tag DbTF

   * python3 AnnotateDbTF.py --input {train_abstracts, train_sentences, test_abstracts, test_sentences}

# TASK 1 - Triage of abstracts with TRI

4. Merge entities, preprocess data, perform feature selection, concatenate features to text and export training sets 

   python3 ExportData.py --type train --data pubtator_train --EXTRACT_entities EXTRACT_train --GNORMPLUS_entities GNORMPLUS_train

   python3 ExportData.py --type test --data pubtator_test --EXTRACT_entities EXTRACT_test --GNORMPLUS_entities GNORMPLUS_test

5. Perform Byte-Pair Encoding of training set

   bash BPE.sh

6. Traing GloVe embeddings of training set

   bash glove.sh

7. Run RNN, CNN and HAN

   bash_RNN.sh

   bash_CNN.sh

   bash_HAN.sh

# TASK 2 - Extraction of TRI interactions at sentence level

