Author: Fabio Curi Paixao
Date: 15.07.2019
fcuri91@gmail.com

# Installations:

1. pip3 install -r requirements.txt

2. Download GNormPlus (both Perl and Java's version) and install them correctly: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/

# Extraction of TRI interactions at the sentence level

## Training Data: 

   * RE/train

## Retrieve and do NER on test data:

   * ´bash retrieve_test.sh´
   * ´bash annotate_test.sh´
   * python3 export_abstracts.py --folder test
   * ´bash minfner_test_gnormplus.sh´
   * ´bash minfner_test.sh´
   * mv ./test/text/*.minfner ./test/NTNU
   * python3 merge_ner.py --folder test
   * python3 filter_words.py

## Build train and test ML data

   If NER has been changed, run ´bash relabel_train.sh´

   * python3 build_data.py --folder train
   * python3 build_data.py --folder test/merged

## Run RE and predict test

   * python3 RunRNN.py --max_num_words 500 --dim_LSTM 100 --attention Att --optimizer adam --oversampling ROS

## Normalize all genes and compare metrics with silver standard

   * bash preprocess_dictionaries.sh
   * python3 build_metric.py
