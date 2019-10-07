#!/bin/bash
echo "Preparing data for abstract classification..."
cd scripts && python3 export_pmids_triage.py
echo "Retrieving abstracts in Pubtator format..."
cd ../bash && mv retrieve_train_triage.sh ../GNormPlusPerl && cd .. && mv train_pmids_triage.txt GNormPlusPerl && cd GNormPlusPerl && ./retrieve_train_triage.sh
echo "Annotating abstracts in Pubtator format..."
cd ../bash && mv annotate_train_triage.sh ../GNormPlusJava && cd ../GNormPlusJava && ./annotate_train_triage.sh
echo "Exporting abstracts as .txt files..."
cd ../scripts && python3 export_abstracts.py --folder train_triage
echo "Annotating abstracts in BRAT format..."
cd ../bash && ./minfner_train_triage_gnormplus.sh && ./minfner_train_triage.sh
echo "Merging all annotations in BRAT format..."
cd ../scripts && python3 merge_ner.py --folder train_triage && python3 filter_words.py --folder train_triage
echo "Exporting triage data..."
python3 build_data_triage.py --folder train_triage
echo "Running byte-pair encoding..."
cd .. && mv triage_data_preprocessed_train.txt subword-nmt-master && mv triage_data_original_train.txt subword-nmt-master && cd bash && mv triage_BPE.sh ../subword-nmt-master
cd ../subword-nmt-master && ./triage_BPE.sh
echo "Training GloVe embeddings..."
mv triage_data_original_train.txt ../glove && mv triage_data_original_train_BPE.txt ../glove && mv triage_data_preprocessed_train.txt ../glove && mv triage_data_preprocessed_train_BPE.txt ../glove
cd ../bash && mv triage_glove.sh ../glove && cd ../glove && ./triage_glove.sh && cd ../scripts
echo "Training RNN. Textual embedded features: presence (or not) of DBTFs and experimental methods."
data=("triage_data_original_train" "triage_data_original_train_BPE" "triage_data_preprocessed_train" "triage_data_preprocessed_train_BPE")
for i in ${data[@]}; do
    python3 RNN_triage.py --train ${i} --test none --report no
done
