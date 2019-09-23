Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

Date: 26.08.2019

# Installations:

Note: Start your venv and unzip all folders into the root folder.

* pip3 install -r requirements.txt
* Install GNormPlus (both Perl and Java's version): https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
* Install GloVe: https://nlp.stanford.edu/projects/glove/
* Clone the Byte-pair encoding repository: https://github.com/rsennrich/subword-nmt
* Download data: https://drive.google.com/open?id=1yMOO2DDF_Jc864mvGq0LIdQtXQuvkQhE

## STEP 1: Prepare data for Triage

   * prepare_triage.sh

## STEP 1: Prepare data for Relation Extraction

   * mv retrieve_test.sh GNormPlusPerl && mv test.txt GNormPlusPerl && cd GNormPlusPerl && bash retrieve_test.sh
   * mv annotate_test.sh GNormPlusJava && cd GNormPlusJava && bash annotate_test.sh
   * python3 export_abstracts.py --folder test
   * bash minfner_test_gnormplus.sh && bash minfner_test.sh
   * python3 merge_ner.py --folder test && python3 filter_words.py

## Preprocess and build train and test ML data

   * python3 build_data_re.py --folder train && python3 build_data_re.py --folder test/merged && python3 preprocessing.py
   
## Normalize all genes

   * bash preprocess_dictionaries.sh
   
## Train and score

   * run.sh
   * OPTION 1: RF with trigger word features and TF-IDF or BoW.
   * OPTION 2: SVM with trigger word features and TF-IDF or BoW.
   * OPTION 3: RNN with word embeddings. Note: If you wish to train your own word embeddings, you can use GloVe.
