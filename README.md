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
|RF_re_train_original_TF-IDF|0.7207|0.6767|0.6875|
|RF_re_train_original_BoW|0.7207|0.6767|0.6875|
|RF_re_train_original_BPE_TF-IDF|0.7146|0.6666|0.6781|
|RF_re_train_original_BPE_BoW|0.7146|0.6666|0.6781|
|RF_re_train_preprocessed_TF-IDF|0.7536|0.6804|0.7034|
|RF_re_train_preprocessed_BoW|0.7536|0.6804|0.7034|
|RF_re_train_preprocessed_BPE_TF-IDF|0.7695|0.6839|0.7097|
|RF_re_train_preprocessed_BPE_BoW|0.7695|0.6839|0.7097|
|SVM_re_train_original_TF-IDF|0.7254|0.6897|0.7018|
|SVM_re_train_original_BoW|0.7254|0.6897|0.7018|
|SVM_re_train_original_BPE_TF-IDF|0.7340|0.7026|0.7149|
|SVM_re_train_original_BPE_BoW|0.7340|0.7026|0.7149|
|SVM_re_train_preprocessed_TF-IDF|0.7325|0.6939|0.7077|
|SVM_re_train_preprocessed_BoW|0.7325|0.6939|0.7077|
|SVM_re_train_preprocessed_BPE_TF-IDF|0.7482|0.7178|0.7297|
|SVM_re_train_preprocessed_BPE_BoW|0.7482|0.7178|0.7297|
|RNN_re_train_original|0.7343|0.7397|0.7333|
|RNN_re_train_original_BPE|0.7302|0.7421|0.7314|
|*RNN_re_train_preprocessed|0.7349|0.7418|0.7353|
|RNN_re_train_preprocessed_BPE|0.7325|0.7406|0.7340|

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
