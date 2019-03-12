Author: Fabio Curi Paixao

# TASK 1 - Detection of TRI interactions in abstracts

## Data : 
data/abstracts.all.labeled.csv

data/hackaton_1.tsv

data/hackaton_2.tsv

## Pipeline

1. Obtain list of PMIDs (ExportPMIDs.py)

2. Download GNormPlus (Perl version) and install it correctly: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/

3. Retrieve original abstracts in PubTator format (retrieve_abstracts.sh)

4. Annotate entities with the EXTRACT tool (https://extract.jensenlab.org/) and export final dataset (EXTRACT.py)

5. Build features, preprocess data, perform feature selection, concatenate features to text and export training sets (BuildTrainingData.py)

6. Train and apply Byte-Pair Encoding (BPE.sh)

7. Download GloVe and install it correctly: https://nlp.stanford.edu/projects/glove/

8. Train GloVe embeddings (glove.sh)

9. Run models (bash_RNN.sh, bash_CNN.sh, bash_HAN.sh) and perform model selection

# TASK 2 - Extraction of TRI interactions at sentence level

## Contextualization (?)

