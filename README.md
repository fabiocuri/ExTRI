Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 27.09.2019

# Installations:

Note: Start your venv and unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
* Install GloVe: https://nlp.stanford.edu/projects/glove/
* Clone the Byte-pair encoding repository: https://github.com/rsennrich/subword-nmt
* Download data: https://drive.google.com/file/d/1bnNKFUwPY0rwh5mHIk40pCfoc9z30R5j/view?usp=sharing

## Run Triage and Relation Extractor

   * bash triage.sh && bash re.sh

### RNN Training results (model/precision/recall/f1-score) of 2-class triage classification (10-fold averaged):
### (*: best model)

triage_data_original_train re_RNN_predictions_RNN_500_100	0.8172899496916509	0.873464739593012	0.8417636310310135

triage_data_original_BPE_train re_RNN_predictions_RNN_500_100	0.8286848109642235	0.8771871062833562	0.8473977599073972

*triage_data_preprocessed_train re_RNN_predictions_RNN_500_100	0.8782606876140656	0.8934898051687348	0.8848864125691396

triage_data_preprocessed_BPE_train re_RNN_predictions_RNN_500_100	0.8200426546207111	0.8751892961791251	0.8422448031770753

triage_data_original_train re_RNN_predictions_RNN_500_500	0.8389016610365303	0.883624387676662	0.8587850286111414

triage_data_original_BPE_train re_RNN_predictions_RNN_500_500	0.840492138368915	0.8592619840972443	0.8486416883577936

triage_data_preprocessed_train re_RNN_predictions_RNN_500_500	0.8565220245298579	0.8972283371267343	0.8713344724368632

triage_data_preprocessed_BPE_train re_RNN_predictions_RNN_500_500	0.8369729262717879	0.8713734105075261	0.8523893330790683

### RF/SVM/RNN Training results (model/precision/recall/f1-score) of 4-class relation extraction classification (10-fold averaged):

re_RF_predictions_original_TF-IDF	0.6903962978845054	0.5989969604100038	0.6236378942237686

re_RF_predictions_original_BoW	0.6939590930711909	0.6024125534995101	0.6294466996249604

re_RF_predictions_preprocessed_TF-IDF	0.7394796380090497	0.5908646941255637	0.6298070728691514

*re_RF_predictions_preprocessed_BoW	0.76028237514518	0.6271982087199479	0.6672203459788648

re_SVM_predictions_original_TF-IDF	0.6667149232688768	0.5732938803590976	0.6025314943302895

re_SVM_predictions_original_BoW	0.6445524342282969	0.6105830806917762	0.62405471441176

re_SVM_predictions_preprocessed_TF-IDF	0.6868570525632485	0.6014914819262646	0.6314622763806066

re_SVM_predictions_preprocessed_BoW	0.6784873008065333	0.6479366370670718	0.6606837148847652

re_RNN_predictions_RNN_500_100_original	0.6651991482079803	0.6687302732202188	0.66153251496263

re_RNN_predictions_RNN_500_100_preprocessed	0.6606784421317221	0.6565454443845071	0.6519647821159091

## Score best models

   * bash score.sh 

### Triage score: PRECISION = 0.8601576404780066, RECALL = 0.3323509185578151, F1-SCORE = 0.4614650115945983

### Comment: the low recall is likely to be due to lack of representative data. Idea: add more sentences to the training corpus.

### Output file: triage.output

### Triage score: PRECISION = 0.2976730129948625, RECALL = 0.07320620899072049, F1-SCORE = 0.10320543044672459

### Comment: add more training sentences.

### Output file: re.output
