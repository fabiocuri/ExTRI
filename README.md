# ExTRI

Detection of TRI interactions through Deep Learning.

Author: Fabio Curi Paixao
Date: 05.03.2019

## Pipeline to train Deep Learning classifiers to detect TRI interactions 


### Export dataset with features 

1. Obtain list of annotated PMIDs (ExportPMIDs.py)

2. Download GNormPlus (Perl version) and install it correctly: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/

3. Retrieve original abstracts in PubTator format (retrieve_abstracts.sh)

4. Annotate entities with the EXTRACT tool and export final dataset (EXTRACT.py)

5. Build features (BuildFeaturesEXTRACT.py)


### Preprocess data 

1. Build and preprocess data, concatenate features to text and export training sets (BuildPreprocessExportTrainingData.py)

2. Train and apply Byte-Pair Encoding (BPE.sh)

3. Train GloVe models of size 100 (glove.sh)


### Run classification

1. Run models (RunRNN.py, RunCNN.py, RunHAN.py) 

