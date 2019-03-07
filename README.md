Author: Fabio Curi Paixao

# TASK 1 - Detection of TRI interactions in abstracts.

## Data : abstracts.all.labeled.csv, Gold_Label_train.tsv, Gold_Label_test.tsv

## Pipeline

### Export dataset with features 

1. Obtain list of annotated PMIDs (ExportPMIDs.py)

2. Download GNormPlus (Perl version) and install it correctly: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/

3. Retrieve original abstracts in PubTator format (retrieve_abstracts.sh)

4. Annotate entities with the EXTRACT tool (https://extract.jensenlab.org/) and export final dataset (EXTRACT.py)

5. Build features (BuildFeaturesEXTRACT.py)


### Preprocess data 

1. Build and preprocess data, perform feature selection, concatenate features to text and export training sets (PreprocessExportTrainingData.py)

2. Train and apply Byte-Pair Encoding (BPE.sh)

3. Download GloVe and install it correctly: https://nlp.stanford.edu/projects/glove/

4. Train GloVe embeddings (glove.sh)


### Run classification

1. Run model selection (bash_RNN.sh, bash_CNN.sh, bash_HAN.sh) 

# TASK 2 - Extraction of TRI interactions at sentence level.
