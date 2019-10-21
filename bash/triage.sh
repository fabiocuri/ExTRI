#!/bin/bash
main_folder=/home/bsclife018/Desktop/ExTRI-master #main_folder is your root folder where the directory has been cloned
script_dir=${main_folder}/scripts
bash_dir=${main_folder}/bash
data_dir=${main_folder}/data
bpe_dir=${main_folder}/subword-nmt-master/subword_nmt
gn_perl=${main_folder}/GNormPlusPerl
gn_java=${main_folder}/GNormPlusJava
glove_dir=${main_folder}/GloVe-master
triage_folder=triage #to use another folder, change this line and set the same name in retrieve_triage.sh and annotate_triage.sh
echo "STEP 1/11: Copying scripts to their respective folders..."
cp ${bash_dir}/retrieve_triage.sh ${gn_perl}
cp ${bash_dir}/annotate_triage.sh ${gn_java}
cp ${bash_dir}/triage_BPE.sh ${bpe_dir}
cp ${bash_dir}/triage_glove.sh ${glove_dir}
echo "STEP 2/11: Preparing data for abstract classification..."
python3 ${script_dir}/export_pmids_triage.py --i ${main_folder}/data --o ${main_folder}
echo "STEP 3/11: Retrieving abstracts in Pubtator format..."
cp ${main_folder}/triage_train_pmids.txt ${gn_perl}
cd ${gn_perl}
bash ${gn_perl}/retrieve_triage.sh
echo "STEP 4/11: Annotating abstracts in Pubtator format..."
cd ${gn_java}
bash ${gn_java}/annotate_triage.sh
echo "STEP 5/11: Exporting abstracts..."
python3 ${script_dir}/export_abstracts.py --i ${main_folder}/${triage_folder}/pubtator --o ${main_folder}/${triage_folder}/text
echo "STEP 6/11: Annotating abstracts in BRAT format..."
bash ${bash_dir}/minfner_triage_gnormplus.sh
bash ${bash_dir}/minfner_triage_ntnu.sh
echo "STEP 7/11: Merging all annotations in BRAT format..."
python3 ${script_dir}/merge_ner.py --i1 ${main_folder}/${triage_folder}/NTNU --i2 ${main_folder}/${triage_folder}/GNormPlus --i3 ${main_folder}/${triage_folder}/text --o ${main_folder}/${triage_folder}/merged
python3 ${script_dir}/filter_words.py --i1 ${data_dir} --i2 ${main_folder}/${triage_folder}/merged
echo "STEP 8/11: Exporting triage data..."
python3 ${script_dir}/build_data_triage.py --i1 ${main_folder}/${triage_folder}/merged --i2 ${main_folder} --i3 ${data_dir} --option train --o ${main_folder}
echo "STEP 9/11: Running byte-pair encoding..."
bash ${bpe_dir}/triage_BPE.sh
echo "STEP 10/11: Training GloVe embeddings..."
cp ${main_folder}/triage_train_original.txt ${main_folder}/triage_train_original_BPE.txt ${main_folder}/triage_train_preprocessed.txt ${main_folder}/triage_train_preprocessed_BPE.txt ${glove_dir}
cd ${glove_dir}
bash ${glove_dir}/triage_glove.sh
echo "STEP 11/11: Training RNN with 10-fold cross-validation. Textual embedded features: presence (or not) of DBTFs and experimental methods."
data=(${main_folder}/triage_train_original.txt ${main_folder}/triage_train_original_BPE.txt ${main_folder}/triage_train_preprocessed.txt ${main_folder}/triage_train_preprocessed_BPE.txt)
for i in ${data[@]}; do
    echo "RNN" ${i}
    python3 ${script_dir}/RNN_triage.py --train ${i} --test none --report no --labels ${main_folder}/triage_train_labels.txt --glove ${glove_dir} --o ${main_folder}
done
