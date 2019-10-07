Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 07.10.2019

### Installations:

Note: Start your venv and unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
* Install GloVe: https://nlp.stanford.edu/projects/glove/
* Clone the Byte-pair encoding repository: https://github.com/rsennrich/subword-nmt
* Download data: https://drive.google.com/file/d/1bnNKFUwPY0rwh5mHIk40pCfoc9z30R5j/view?usp=sharing

### Train Triage and Relation Extractor

   * chmod -R 777 ./
   * ./bash/triage.sh && ./bash/re.sh

### RNN Training results of 2-class text classification (10-fold averaged):

|Model |Precision|Recall|F1-score|
|-------------|-------------|-------------|-------------|
|triage_RNN_original|0.8498|0.8724|0.8583|
|triage_RNN_original_BPE|0.8493|0.8563|0.8499|
|*triage_RNN_preprocessed|0.8576|0.8925|0.8733|
|triage_RNN_preprocessed_BPE|0.8395|0.8878|0.8608|

### RF/SVM/RNN Training results of 4-class relation extraction classification (10-fold averaged):

|Model |Precision|Recall|F1-score|
|-------------|-------------|-------------|-------------|
|re_RF_original_TF-IDF|0.6903|0.5989|0.6236|
|re_RF_original_BoW|0.6939|0.6024|0.6294|
|re_RF_preprocessed_TF-IDF|0.7394|0.5908|0.6298|
|*re_RF_preprocessed_BoW|0.7602|0.6271|0.6672|
|re_SVM_original_TF-IDF|0.6667|0.5732|0.6025|
|re_SVM_original_BoW|0.6445|0.6105|0.6240|
|re_SVM_preprocessed_TF-IDF|0.6868|0.6014|0.6314|
|re_SVM_preprocessed_BoW|0.6784|0.6479|0.6606|
|re_RNN_original|0.6670|0.6680|0.6625|
|re_RNN_preprocessed|0.6581|0.6519|0.6498|

### Predict and score with best models. Reference: silver standard w/ high confidence 

   * ./bash/score.sh 

|Task |Precision|Recall|F1-score|Output file|Comment|
|-------------|-------------|-------------|-------------|-------------|-------------|
|Triage|0.6382|0.7416|0.6860|triage.output|List of positive PMIDs|
|RE score 1|0.6310|0.6857|0.6572|re_pmids.output|List of positive PMIDs|
|RE score 2|0.1589|0.2492|0.1941|re_sentences.output|Positive sentences with relations|

##### Comments: 

##### 1: The triage classification scores higher in the task of classifying text as having TRI or not.

##### 2: The low scores in the relations extracted are most likely due to a lack of representativity of the training data.

##### Idea: add more sentences to the training corpus.
