python3 export_pmids_triage.py
mv retrieve_train_triage.sh GNormPlusPerl
mv train_pmids_triage.txt GNormPlusPerl
cd GNormPlusPerl
bash retrieve_train_triage.sh
cd ..
mv annotate_train_triage.sh GNormPlusJava
cd GNormPlusJava
bash annotate_train_triage.sh
echo "PMIDs have been annotated!"
python3 export_abstracts.py --folder train_triage
bash minfner_train_triage_gnormplus.sh 
bash minfner_train_triage.sh 
python3 merge_ner.py --folder train_triage 
python3 filter_words.py --folder train_triage
python3 build_data_triage.py --input train_triage
mv triage_data_preprocessed.txt subword-nmt-master && mv triage_data_original.txt subword-nmt-master && mv triage_BPE.sh subword-nmt-master
cd subword-nmt-master && bash triage_BPE.sh
mv triage_data_original.txt ../glove && mv triage_data_original_BPE.txt ../glove && mv triage_data_preprocessed.txt ../glove && mv triage_data_preprocessed_BPE.txt ../glove
mv triage_glove.sh glove && cd glove && bash triage_glove.sh && cd ..
