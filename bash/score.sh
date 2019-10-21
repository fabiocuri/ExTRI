#!/bin/bash
main_folder=/home/bsclife018/Desktop/ExTRI-master #main_folder is your root folder where the directory has been cloned
script_dir=${main_folder}/scripts
data_dir=${main_folder}/data
models_dir=${main_folder}/models
glove_dir=${main_folder}/GloVe-master
re_folder=re
rm -r ${models_dir}/*
echo "STEP 1/3: Running and scoring best Triage model..."
python3 ${script_dir}/build_data_triage.py --i1 ${main_folder}/${re_folder}/merged --i2 ${main_folder} --i3 ${data_dir} --option test --o ${main_folder}
python3 ${script_dir}/RNN_triage.py --train ${main_folder}/triage_train_preprocessed.txt --test ${main_folder}/triage_test_preprocessed.txt --report yes --labels ${main_folder}/triage_train_labels.txt --glove ${glove_dir} --o ${main_folder}
python3 ${script_dir}/score.py --task triage --predictions ${models_dir}/triage_predictions --i1 ${data_dir} --i2 ${main_folder}
echo "STEP 2/3: Running and scoring best RE model..."
python3 ${script_dir}/RNN_re.py --train ${main_folder}/re_train_preprocessed.txt --test ${main_folder}/re_test_preprocessed.txt --report yes --labels ${main_folder}/re_train_labels.txt --glove ${glove_dir} --o ${main_folder}
python3 ${script_dir}/score.py --task re --predictions ${models_dir}/re_predictions --i1 ${data_dir} --i2 ${main_folder}
echo "STEP 3/3: Removing files..."
mv ${main_folder}/requirements.txt ${data_dir} && rm -r ${main_folder}/*.txt && rm -r ${main_folder}/*.BPE && rm -r ${main_folder}/*.csv && mv ${data_dir}/requirements.txt ${main_folder}
