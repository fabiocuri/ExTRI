#!/bin/bash
echo "Retrieving abstracts in Pubtator format..."
cd bash && mv retrieve_test.sh ../GNormPlusPerl && mv ../data/test.txt ../GNormPlusPerl && cd ../GNormPlusPerl && ./retrieve_test.sh
echo "Annotating abstracts in Pubtator format..."
cd ../bash && mv annotate_test.sh ../GNormPlusJava && cd ../GNormPlusJava && ./annotate_test.sh
echo "Exporting abstracts as .txt files..."
cd ../scripts && python3 export_abstracts.py --folder test
echo "Annotating abstracts in BRAT format..."
cd ../bash && ./minfner_test_gnormplus.sh && ./minfner_test.sh
echo "Merging all annotations in BRAT format..."
cd ../scripts && python3 merge_ner.py --folder test && python3 filter_words.py --folder test
echo "Exporting relation extraction data..."
python3 build_data_re.py --folder data/train && python3 build_data_re.py --folder test/merged
echo "Running byte-pair encoding..."
python ./subword-nmt-master/apply_bpe.py -c ./subword-nmt-master/triage_data_original_train.BPE < ./re_test_original.txt > ./re_test_original_BPE.txt
python ./subword-nmt-master/apply_bpe.py -c ./subword-nmt-master/triage_data_preprocessed_train.BPE < ./re_test_preprocessed.txt > ./re_test_preprocessed_BPE.txt
echo "Training Random Forests. Features: presence (or not) of keywords."
cd scripts && python3 RF.py --report no
echo "Training SVM. Features: presence (or not) of keywords."
python3 SVM.py --report no
echo "Training RNN."
python3 RNN_re.py --train original --report no
python3 RNN_re.py --train preprocessed --report no
