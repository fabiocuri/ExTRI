Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 27.09.2019

# Installations:

Note: Start your venv and unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
* Install GloVe: https://nlp.stanford.edu/projects/glove/
* Clone the Byte-pair encoding repository: https://github.com/rsennrich/subword-nmt
* Download data: https://drive.google.com/file/d/1gkra2bkqJXpSPEtmoK0at09Lf3O9VHg_/view?usp=sharing

## STEP 1: Prepare data for Triage and Relation Extraction

   * prepare_triage.sh && prepare_re.sh

## STEP 2: Run Triage classifier

   * bash triage.sh

RNN Training results of 2-class classification(10-fold averaged):
model/precision/recall/f1-score
(*: best model)

triage_data_original RNN_500_100_Att_adam_ROS	0.8389178114493057	0.8849179651255275	0.858478896163508
triage_data_original_BPE RNN_500_100_Att_adam_ROS	0.8322353942190691	0.8545545342307111	0.8405640864569198
*triage_data_preprocessed RNN_500_100_Att_adam_ROS	0.8604594447424571	0.8947687779485536	0.8755859014962933
triage_data_preprocessed_BPE RNN_500_100_Att_adam_ROS	0.8384581351814943	0.9004145439315234	0.8654619401459156
triage_data_original RNN_500_500_Att_adam_ROS	0.8510718830630231	0.8644009950615787	0.8550819379362637
triage_data_original_BPE RNN_500_500_Att_adam_ROS	0.8252016892639402	0.870820090874415	0.8453483797192888
triage_data_preprocessed RNN_500_500_Att_adam_ROS	0.8561052975344314	0.8822839967161163	0.8665894519176529
triage_data_preprocessed_BPE RNN_500_500_Att_adam_ROS	0.84060149495536	0.8962430682878091	0.8629526929408377

## STEP 3: Run Relation Extraction

   * bash re.sh

SVM/RF/RNN Training results of 4-class classification (10-fold averaged):
model/precision/recall/f1-score
(*: best model)

re_RF_predictions_original_TF-IDF	0.6903962978845054	0.5989969604100038	0.6236378942237686
re_RF_predictions_original_BoW	0.6939590930711909	0.6024125534995101	0.6294466996249604
re_RF_predictions_preprocessed_TF-IDF	0.7394796380090497	0.5908646941255637	0.6298070728691514
re_RF_predictions_preprocessed_BoW	0.76028237514518	0.6271982087199479	0.6672203459788648
re_SVM_predictions_original_TF-IDF	0.6667149232688768	0.5732938803590976	0.6025314943302895
re_SVM_predictions_original_BoW	0.6445524342282969	0.6105830806917762	0.62405471441176
re_SVM_predictions_preprocessed_TF-IDF	0.6868570525632485	0.6014914819262646	0.6314622763806066
re_SVM_predictions_preprocessed_BoW	0.6784873008065333	0.6479366370670718	0.6606837148847652
*re_RNN_predictions_RNN_500_100_original	0.6738366908142693	0.6771629414593814	0.6693733170575776
re_RNN_predictions_RNN_500_100_preprocessed	0.6521526027847294	0.6520864173562056	0.646751208233448
