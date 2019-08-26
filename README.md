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

## Build train and test ML data

   * python3 build_data.py --folder train && python3 build_data.py --folder test/merged

## Run RNN

Note: If you wish to train your own word embeddings, replace 'vectors_train_positive_sentences.txt' by the new vectors.

   * python3 RunRNN.py --max_num_words 500 --dim_LSTM 100 --attention Att --optimizer adam --oversampling ROS

## Normalize all genes and compare metrics with silver standard

   * bash preprocess_dictionaries.sh
   * python3 build_metric.py
