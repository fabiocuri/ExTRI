#!/bin/bash
main_folder=/home/bsclife018/Desktop/ExTRI-master #main_folder is your root folder where the directory has been cloned
script_dir=${main_folder}/scripts
bash_dir=${main_folder}/bash
data_dir=${main_folder}/data
bpe_dir=${main_folder}/subword-nmt-master/subword_nmt
gn_perl=${main_folder}/GNormPlusPerl
gn_java=${main_folder}/GNormPlusJava
glove_dir=${main_folder}/GloVe-master
re_folder=re #to use another folder, change this line and set the same name in retrieve_re.sh and annotate_re.sh
echo "STEP 1/10: Copying scripts to their respective folders..."
cp ${bash_dir}/retrieve_re.sh ${data_dir}/re.txt ${gn_perl}
cp ${bash_dir}/annotate_re.sh ${gn_java}
cp ${bash_dir}/re_BPE.sh ${bpe_dir}
cp ${bash_dir}/re_glove.sh ${glove_dir}
echo "STEP 2/10: Retrieving abstracts in Pubtator format..."
cd ${gn_perl}
bash ${gn_perl}/retrieve_re.sh
echo "STEP 3/10: Annotating abstracts in Pubtator format..."
cd ${gn_java}
bash ${gn_java}/annotate_re.sh
echo "STEP 4/10: Exporting abstracts..."
python3 ${script_dir}/export_abstracts.py --i ${main_folder}/${re_folder}/pubtator --o ${main_folder}/${re_folder}/text
echo "STEP 5/10: Annotating abstracts in BRAT format..."
bash ${bash_dir}/minfner_re_gnormplus.sh
bash ${bash_dir}/minfner_re_ntnu.sh
echo "STEP 6/10: Merging all annotations in BRAT format..."
python3 ${script_dir}/merge_ner.py --i1 ${main_folder}/${re_folder}/NTNU --i2 ${main_folder}/${re_folder}/GNormPlus --i3 ${main_folder}/${re_folder}/text --o ${main_folder}/${re_folder}/merged
python3 ${script_dir}/filter_words.py --i1 ${data_dir} --i2 ${main_folder}/${re_folder}/merged
echo "STEP 7/10: Exporting relation extraction data..."
python3 ${script_dir}/build_data_re.py --i1 ${data_dir}/train --i2 ${data_dir} --option train --o ${main_folder}
python3 ${script_dir}/build_data_re.py --i1 ${main_folder}/${re_folder}/merged --i2 ${data_dir} --option test --o ${main_folder}
echo "STEP 8/10: Running byte-pair encoding..."
bash ${bpe_dir}/re_BPE.sh
echo "STEP 9/10: Training GloVe embeddings..."
cp ${main_folder}/re_train_original.txt ${main_folder}/re_train_original_BPE.txt ${main_folder}/re_train_preprocessed.txt ${main_folder}/re_train_preprocessed_BPE.txt ${glove_dir}
cd ${glove_dir}
bash ${glove_dir}/re_glove.sh
echo "STEP 10/10: Training RF, SVM and RNN with 10-fold cross-validation. Textual embedded features: presence (or not) of DBTFs and experimental methods."
data=(${main_folder}/re_train_original.txt ${main_folder}/re_train_original_BPE.txt ${main_folder}/re_train_preprocessed.txt ${main_folder}/re_train_preprocessed_BPE.txt)
f=("TF-IDF" "BoW")
echo "Additional features for RF and SVM: presence (or not) of keywords."
for i in ${data[@]}; do
    for j in ${f[@]}; do
        echo "RF" ${i} ${j}
        python3 ${script_dir}/RF.py --train ${i} --test none --labels ${main_folder}/re_train_labels.txt --features ${f} --report no --dictionary ${data_dir} --o ${main_folder}
        echo "SVM" ${i} ${j}
        python3 ${script_dir}/SVM.py --train ${i} --test none --labels ${main_folder}/re_train_labels.txt --features ${f} --report no --dictionary ${data_dir} --o ${main_folder}
    done
    echo "RNN" ${i}
    python3 ${script_dir}/RNN_re.py --train ${i} --test none --report no --labels ${main_folder}/re_train_labels.txt --glove ${glove_dir} --o ${main_folder}
done
