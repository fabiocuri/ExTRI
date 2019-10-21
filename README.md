Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 07.10.2019

### Installations:

Note: Start your venv and unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
* Install GloVe: https://github.com/stanfordnlp/GloVe
* Clone the Byte-pair encoding repository: https://github.com/rsennrich/subword-nmt
* Download data: https://drive.google.com/file/d/1bnNKFUwPY0rwh5mHIk40pCfoc9z30R5j/view?usp=sharing

### Train Triage and Relation Extractor

   * chmod -R 777 ./
   * ./bash/triage.sh
   * ./bash/re.sh

### RNN Training results of 2-class text classification (10-fold averaged):

|Model|Precision|Recall|F1-score|
|-------------|-------------|-------------|-------------|
|triage_train_original|0.8552|0.8842|0.8670|
|triage_train_original_BPE|0.8347|0.8375|0.8346|
|*triage_train_preprocessed|0.8622|0.8955|0.8773|
|triage_train_preprocessed_BPE|0.8516|0.8763|0.8607|

### RF/SVM/RNN Training results of 4-class relation extraction classification (10-fold averaged):

|Model|Precision|Recall|F1-score|
|RF|||
|re_train_original_TF-IDF|0.7207|0.6767|0.6875|
|re_train_original_BoW|0.7207|0.6767|0.6875|

### Predict and score with best models. Reference: silver standard w/ high confidence 

   * ./bash/score.sh 

|Task|Precision|Recall|F1-score|Output file|Description|
|-------------|-------------|-------------|-------------|-------------|-------------|
|Triage|0.6336|0.7854|0.7014|triage_pmids.output|List of positive PMIDs|
|RE score 1|0.6097|0.8448|0.7083|re_pmids.output|List of positive PMIDs|
|RE score 2|0.1436|0.6219|0.2334|re_sentences.output|Positive sentences with relations|

##### Comments: 

##### 1: The relation extraction scores higher in the task of triage.

##### 2: The low scores in the relations extracted are most likely due to a lack of representativity of the training data.

##### Idea: add more sentences to the training corpus.
