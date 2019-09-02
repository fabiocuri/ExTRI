Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 26.08.2019

# Installations:

Note: You must unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
* Install GloVe: https://nlp.stanford.edu/projects/glove/
* Download data: https://drive.google.com/open?id=1yMOO2DDF_Jc864mvGq0LIdQtXQuvkQhE.

## Retrieve and do NER on test data:

   * mv retrieve_test.sh GNormPlusPerl && mv test.txt GNormPlusPerl && cd GNormPlusPerl && bash retrieve_test.sh
   * mv annotate_test.sh GNormPlusJava && cd GNormPlusJava && bash annotate_test.sh
   * python3 export_abstracts.py --folder test
   * bash minfner_test_gnormplus.sh && bash minfner_test.sh
   * python3 merge_ner.py --folder test && python3 filter_words.py

## Preprocess and build train and test ML data

   * python3 build_data.py --folder train && python3 build_data.py --folder test/merged && python3 preprocessing.py
   
## Normalize all genes

   * bash preprocess_dictionaries.sh
   
## Train and score

   * run.sh
   * OPTION 1: SVM with trigger word features and TF-IDF or BoW.
   * OPTION 2: RNN with word embeddings. Note: If you wish to train your own word embeddings, you can use GloVe.
   
## Results on silver standard (Florian high confidence):

###### RF_predictions_preprocessed_TF-IDF: 
###### PRECISION = 0.30025667440764087, RECALL =  : 0.2634554396286253, F1-SCORE =  : 0.21149080354452487
###### RF_predictions_preprocessed_BPE_TF-IDF: 
###### PRECISION = 0.3009844015779378, RECALL =  : 0.2548298631857946, F1-SCORE =  : 0.20900717507947877
###### RF_predictions_preprocessed_shortest_TF-IDF: 
###### PRECISION = 0.3553184355338159, RECALL =  : 0.2760031254308958, F1-SCORE =  : 0.2423683891886074
###### RF_predictions_preprocessed_shortest_BPE_TF-IDF: 
###### PRECISION = 0.35743964765030956, RECALL =  : 0.2635933262857931, F1-SCORE =  : 0.23840536533318551
###### RF_predictions_preprocessed_BoW: 
###### PRECISION = 0.3003434765348568, RECALL =  : 0.2665961912641142, F1-SCORE =  : 0.2125390854016025
###### RF_predictions_preprocessed_BPE_BoW: 
###### PRECISION = 0.30178336115259596, RECALL =  : 0.26314902483491903, F1-SCORE =  : 0.21214759919715917
###### RF_predictions_preprocessed_shortest_BoW: 
###### PRECISION = 0.36033481313437504, RECALL =  : 0.2822846287018737, F1-SCORE =  : 0.24712967433875208
###### RF_predictions_preprocessed_shortest_BPE_BoW: 
###### PRECISION = 0.358054210846089, RECALL =  : 0.27989459331096506, F1-SCORE =  : 0.2451425044281037
###### SVM_predictions_preprocessed_TF-IDF: 
###### PRECISION = 0.30094803621070637, RECALL =  : 0.25873665180554917, F1-SCORE =  : 0.21029169131152134
###### SVM_predictions_preprocessed_BPE_TF-IDF: 
###### PRECISION = 0.3012791518155995, RECALL =  : 0.25512095723981554, F1-SCORE =  : 0.20924724021588204
###### SVM_predictions_preprocessed_shortest_TF-IDF: 
###### PRECISION = 0.3606832149413361, RECALL =  : 0.2769376905516998, F1-SCORE =  : 0.2452196680390973
###### SVM_predictions_preprocessed_shortest_BPE_TF-IDF: 
###### PRECISION = 0.3579466929911155, RECALL =  : 0.2777650104947067, F1-SCORE =  : 0.24427213505702602
###### SVM_predictions_preprocessed_BoW: 
###### PRECISION = 0.2987592965871968, RECALL =  : 0.2634094774095693, F1-SCORE =  : 0.21073210192800323
###### SVM_predictions_preprocessed_BPE_BoW: 
###### PRECISION = 0.29874180773139875, RECALL =  : 0.26118797015519907, F1-SCORE =  : 0.21000893104616428
###### SVM_predictions_preprocessed_shortest_BoW: 
###### PRECISION = 0.3605033039211807, RECALL =  : 0.27917451854575537, F1-SCORE =  : 0.24600886992797402
###### SVM_predictions_preprocessed_shortest_BPE_BoW: 
###### PRECISION = 0.35965281536271976, RECALL =  : 0.274884711433868, F1-SCORE =  : 0.24393792104851703
###### RNN_500_100_preprocessed_predictions: 
###### PRECISION = 0.32955945641886464, RECALL =  : 0.2827442508924331, F1-SCORE =  : 0.23241024091075094
###### RNN_500_100_preprocessed_BPE_predictions: 
###### PRECISION = 0.3148108640354476, RECALL =  : 0.29389774938334023, F1-SCORE =  : 0.22842615415758702
###### RNN_500_100_preprocessed_shortest_predictions: 
###### PRECISION = 0.39066237550558863, RECALL =  : 0.2752370884466302, F1-SCORE =  : 0.2579732621088758
###### RNN_500_100_preprocessed_shortest_BPE_predictions: 
###### PRECISION = 0.39410451094238497, RECALL =  : 0.2703804139663863, F1-SCORE =  : 0.2572913553428632
###### RNN_500_500_preprocessed_predictions: 
###### PRECISION = 0.34426716692360554, RECALL =  : 0.283969910067258, F1-SCORE =  : 0.24006890566917508
###### RNN_500_500_preprocessed_BPE_predictions: 
###### PRECISION = 0.3283185686079223, RECALL =  : 0.2878766986870126, F1-SCORE =  : 0.2334988163511305
###### RNN_500_500_preprocessed_shortest_predictions: 
###### PRECISION = 0.3859075756585398, RECALL =  : 0.2713609413062463, F1-SCORE =  : 0.25420335407203287
###### RNN_500_500_preprocessed_shortest_BPE_predictions: 
###### PRECISION = 0.4008103653511583, RECALL =  : 0.26825083115012793, F1-SCORE =  : 0.2591430474358026
###### RNN_1000_100_preprocessed_predictions: 
###### PRECISION = 0.32249841815059205, RECALL =  : 0.2733066752462809, F1-SCORE =  : 0.22572155230226112
###### RNN_1000_100_preprocessed_BPE_predictions: 
###### PRECISION = 0.3150671004473363, RECALL =  : 0.28487383370869146, F1-SCORE =  : 0.22578001202120102
###### RNN_1000_100_preprocessed_shortest_predictions: 
###### PRECISION = 0.3902033949485581, RECALL =  : 0.2771675016469795, F1-SCORE =  : 0.25861649964976485
###### RNN_1000_100_preprocessed_shortest_BPE_predictions: 
###### PRECISION = 0.3861835143714773, RECALL =  : 0.2739807877924346, F1-SCORE =  : 0.2554677790317281
###### RNN_1000_500_preprocessed_predictions: 
###### PRECISION = 0.3306793094712461, RECALL =  : 0.2799711970093916, F1-SCORE =  : 0.232019857669771
###### RNN_1000_500_preprocessed_BPE_predictions: 
###### PRECISION = 0.3258876509320559, RECALL =  : 0.2782859156440073, F1-SCORE =  : 0.22908166804346045
###### RNN_1000_500_preprocessed_shortest_predictions: 
###### PRECISION = 0.38943357451727534, RECALL =  : 0.27871489635519603, F1-SCORE =  : 0.2589479527710363
###### RNN_1000_500_preprocessed_shortest_BPE_predictions: 
###### PRECISION = 0.38520913737352663, RECALL =  : 0.2828974582892862, F1-SCORE =  : 0.25883820684628106
