Author: Fabio Curi Paixao

## Data : 

data/abstracts.all.labeled.csv

data/hackaton_1.tsv

data/hackaton_2.tsv

## Installations:

1. pip3 install -r requirements

2. Download GNormPlus (Perl version) and install it correctly: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/

3. Download GloVe and install it correctly: https://nlp.stanford.edu/projects/glove/

## Extract entities through EXTRACTOR.

1. Obtain list of PMIDs 

   python3 ExportPMIDs.py

* Note that if you already have a list of PMIDs, you can skip this step.

2. Retrieve articles in PubTator format 

   bash retrieve_articles.sh

3. Annotate articles with the EXTRACT tool (https://extract.jensenlab.org/) and export *.ann and *.txt files

   python3 EXTRACT.py --input articles_pubtator --output articles_entities

   python3 EXTRACT.py --input articles_pubtator_test --output articles_entities_test

# TASK 1 - Detection of TRI interactions in abstracts

1. Build features, preprocess data, perform feature selection, concatenate features to text and export training sets 

   python3 ExportData.py --type train --data articles_pubtator --entities articles_entities

   python3 ExportData.py --type test --data articles_pubtator_test --entities articles_entities_test

2. Perform Byte-Pair Encoding of training set

   bash BPE.sh

3. Traing GloVe embeddings of training set

   bash glove.sh

## Run Models and perform model selection

4. Run RNN, CNN and HAN

   bash_RNN.sh

   bash_CNN.sh

   bash_HAN.sh

# TASK 2 - Extraction of TRI interactions at sentence level

By now, the entities have already been exported in BRAT format as *.ann files.
