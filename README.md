Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 26.08.2019

# Installations:

Note: You must unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/ and place folders 'GNormPlusPerl' and 'GNormPlusJava' in the root folder.
* Install GloVe and place the folder 'glove' in the root folder: https://nlp.stanford.edu/projects/glove/ 
* Download data: https://drive.google.com/file/d/1d3Q4h6biExbBiSEDG9JmbqO03zsW60mo/view?usp=sharing and place subfolders in the root folder.
* Clone the repository for BPE and place folder in the root folder: https://github.com/rsennrich/subword-nmt

## Retrieve and do NER on test data:

   * mv retrieve_test.sh GNormPlusPerl && mv test.txt GNormPlusPerl && cd GNormPlusPerl && ./retrieve_test.sh
   * cd .. && mv annotate_test.sh GNormPlusJava && cd GNormPlusJava && ./annotate_test.sh
   * cd .. && python3 export_abstracts.py --folder test
   * ./minfner_test_gnormplus.sh && ./minfner_test.sh
   * python3 merge_ner.py --folder test && python3 filter_words.py

## Preprocess, build train and test ML data, apply BPE (byte-pair encoding), train embeddings and normalize all genes

   * python3 build_data.py --folder train && python3 build_data.py --folder test/merged && python3 preprocessing.py && ./BPE.sh && cd .. && ./preprocess_dictionaries.sh
   
## Train and score

   * ./run.sh
   * OPTION 1: RF with trigger word features and TF-IDF or BoW.
   * OPTION 2: SVM with trigger word features and TF-IDF or BoW.
   * OPTION 3: RNN with word embeddings. Note: If you wish to train your own word embeddings, you can use GloVe.
   
## Results on silver standard (Florian high confidence):

RF_predictions_preprocessed_TF-IDF: PRECISION = 0.309174116645381, RECALL = 0.15421612553882744, F1-SCORE = 0.167318469539975
RF_predictions_preprocessed_BPE_TF-IDF: PRECISION = 0.30958295776815353, RECALL = 0.1495126664260081, F1-SCORE = 0.16462666152281416
RF_predictions_preprocessed_shortest_TF-IDF: PRECISION = 0.36221989628431417, RECALL = 0.16166946255282102, F1-SCORE = 0.1867942466341583
RF_predictions_preprocessed_shortest_BPE_TF-IDF: PRECISION = 0.3646752857035685, RECALL = 0.15483192831206335, F1-SCORE = 0.1827661187735382
RF_predictions_preprocessed_BoW: PRECISION = 0.30915367716288317, RECALL = 0.15688106512645192, F1-SCORE = 0.16886857142857142
RF_predictions_preprocessed_BPE_BoW: PRECISION = 0.3106068691698508, RECALL = 0.15449217505786422, F1-SCORE = 0.16790035135493772
RF_predictions_preprocessed_shortest_BoW: PRECISION = 0.369232223903177, RECALL = 0.16584205720595419, F1-SCORE = 0.19145206956990696
RF_predictions_preprocessed_shortest_BPE_BoW: PRECISION = 0.3665752906701957, RECALL = 0.1646953899730321, F1-SCORE = 0.18997464881878193
SVM_predictions_preprocessed_TF-IDF: PRECISION = 0.3133860788438672, RECALL = 0.14550994839997453, F1-SCORE = 0.1632081931584745
SVM_predictions_preprocessed_BPE_TF-IDF: PRECISION = 0.31439792411810086, RECALL = 0.1427919223663814, F1-SCORE = 0.16175259033862327
SVM_predictions_preprocessed_shortest_TF-IDF: PRECISION = 0.3702512065338448, RECALL = 0.15883464633809696, F1-SCORE = 0.18695793446474543
SVM_predictions_preprocessed_shortest_BPE_TF-IDF: PRECISION = 0.37014706995282665, RECALL = 0.15578748433949843, F1-SCORE = 0.1848043074404106
SVM_predictions_preprocessed_BoW: PRECISION = 0.3075985371800081, RECALL = 0.15270847047331876, F1-SCORE = 0.1659695016760808
SVM_predictions_preprocessed_BPE_BoW: PRECISION = 0.3083540223877347, RECALL = 0.15033019769392478, F1-SCORE = 0.16477077672332033
SVM_predictions_preprocessed_shortest_BoW: PRECISION = 0.37161003646550267, RECALL = 0.16337884611301043, F1-SCORE = 0.19042669042669041
SVM_predictions_preprocessed_shortest_BPE_BoW: PRECISION = 0.3696605808111751, RECALL = 0.1601511901981186, F1-SCORE = 0.18771466971974712
RNN_500_100_preprocessed_predictions: PRECISION = 0.3445149592021759, RECALL = 0.1613827957445905, F1-SCORE = 0.18178991305179817
RNN_500_100_preprocessed_BPE_predictions: PRECISION = 0.329084538830256, RECALL = 0.16255069755590001, F1-SCORE = 0.1781040239178232
RNN_500_100_preprocessed_shortest_predictions: PRECISION = 0.4049632151092566, RECALL = 0.15662625018580256, F1-SCORE = 0.19373563595771226
RNN_500_100_preprocessed_shortest_BPE_predictions: PRECISION = 0.4178852643419573, RECALL = 0.1498842715477884, F1-SCORE = 0.19124450495485426
RNN_500_500_preprocessed_predictions: PRECISION = 0.35333411187667874, RECALL = 0.16061835092264243, F1-SCORE = 0.1837170892839794
RNN_500_500_preprocessed_BPE_predictions: PRECISION = 0.33916317091534903, RECALL = 0.16739218142823775, F1-SCORE = 0.18397806173055603
RNN_500_500_preprocessed_shortest_predictions: PRECISION = 0.3920081445660473, RECALL = 0.16352748816172255, F1-SCORE = 0.19574998093591928
RNN_500_500_preprocessed_shortest_BPE_predictions: PRECISION = 0.39044668215853345, RECALL = 0.16055464718748008, F1-SCORE = 0.19322276456006746
RNN_1000_100_preprocessed_predictions: PRECISION = 0.3404578724598368, RECALL = 0.154577113371414, F1-SCORE = 0.17630924052242467
RNN_1000_100_preprocessed_BPE_predictions: PRECISION = 0.32172072183170963, RECALL = 0.16619242774934703, F1-SCORE = 0.17803583919563698
RNN_1000_100_preprocessed_shortest_predictions: PRECISION = 0.4043880461731226, RECALL = 0.16105365977958508, F1-SCORE = 0.19695012302079343
RNN_1000_100_preprocessed_shortest_BPE_predictions: PRECISION = 0.39908295356054463, RECALL = 0.15155118595120293, F1-SCORE = 0.18850283269284102
RNN_1000_500_preprocessed_predictions: PRECISION = 0.36089294120561927, RECALL = 0.15465143439577006, F1-SCORE = 0.18168664479674695
RNN_1000_500_preprocessed_BPE_predictions: PRECISION = 0.3238564135676668, RECALL = 0.17040749155925508, F1-SCORE = 0.18109605424983358
RNN_1000_500_preprocessed_shortest_predictions: PRECISION = 0.3989782683501416, RECALL = 0.16003440001698765, F1-SCORE = 0.19490402208558813
RNN_1000_500_preprocessed_shortest_BPE_predictions: PRECISION = 0.39777428646242474, RECALL = 0.16128724014184698, F1-SCORE = 0.19553982300884956
