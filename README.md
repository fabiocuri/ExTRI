Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 27.09.2019

### Installations:

Note: Start your venv and unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
* Install GloVe: https://nlp.stanford.edu/projects/glove/
* Clone the Byte-pair encoding repository: https://github.com/rsennrich/subword-nmt
* Download data: https://drive.google.com/file/d/1bnNKFUwPY0rwh5mHIk40pCfoc9z30R5j/view?usp=sharing

### Run Triage and Relation Extractor

   * bash triage.sh && bash re.sh

### RNN Training results of 2-class triage classification (10-fold averaged):

| Model  | Precision | Recall | F1-score |
| ------------- | ------------- | ------------- | ------------- |
| triage_data_original_train_re_RNN_predictions_RNN_500_100	| 0.8172	| 0.8734	| 0.8417 |
| triage_data_original_BPE_train_re_RNN_predictions_RNN_500_100 |	0.8286 |	0.8771 |	0.8473 |
| *triage_data_preprocessed_train_re_RNN_predictions_RNN_500_100 |	0.8782 |	0.8934 |	0.8848 |
| triage_data_preprocessed_BPE_train_re_RNN_predictions_RNN_500_100 |	0.8200 |	0.8751 | 0.8422 |
| triage_data_original_train_re_RNN_predictions_RNN_500_500 |	0.8389 | 0.8836 |	0.8587 |
| triage_data_original_BPE_train_re_RNN_predictions_RNN_500_500	| 0.8404 |	0.8592 |	0.8486 |
| triage_data_preprocessed_train_re_RNN_predictions_RNN_500_500	| 0.8565 |	0.8972 |	0.8713 |
| triage_data_preprocessed_BPE_train_re_RNN_predictions_RNN_500_500	| 0.8369 |	0.8713 |	0.8523 |


### RF/SVM/RNN Training results of 4-class relation extraction classification (10-fold averaged):

| Model  | Precision | Recall | F1-score |
| ------------- | ------------- | ------------- | ------------- |
| re_RF_predictions_original_TF-IDF |	0.6903 |	0.5989 |	0.6236 |
| re_RF_predictions_original_BoW |	0.6939 |	0.6024 |	0.62944 |
| re_RF_predictions_preprocessed_TF-IDF |	0.7394 |	0.5908 |	0.6298 |
| *re_RF_predictions_preprocessed_BoW |	0.7602 |	0.6271 |	0.6672 |
| re_SVM_predictions_original_TF-IDF |	0.6667 |	0.5732 |	0.6025 |
| re_SVM_predictions_original_BoW |	0.6445 |	0.6105 |	0.6240 |
| re_SVM_predictions_preprocessed_TF-IDF |	0.6868 |	0.6014 |	0.6314 |
| re_SVM_predictions_preprocessed_BoW |	0.6784 |	0.6479 |	0.6606 |
| re_RNN_predictions_RNN_500_100_original |	0.6651 |	0.6687 |	0.6615 |
| re_RNN_predictions_RNN_500_100_preprocessed |	0.6606 |	0.6565 |	0.6519 |

### Score best models

   * bash score.sh 
   
| Task  | Precision | Recall | F1-score | Output file |
| ------------- | ------------- | ------------- | ------------- |------------- |
| Triage |	0.8601 |	0.3323 |	0.4614 | triage.output |
| RE |	0.2976 |	0.0732 |	0.1032 | re.output |

### Comment: the low recall is likely to be due to lack of representative data. Idea: add more sentences to the training corpus.
