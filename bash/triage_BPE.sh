#!/bin/bash
main_folder=/home/bsclife018/Desktop/ExTRI-master #main_folder is your root folder where the directory has been cloned
bpe_dir=${main_folder}/subword-nmt-master/subword_nmt
DATA=(${main_folder}/triage_train_preprocessed ${main_folder}/triage_train_original)

for NAME in ${DATA[@]}; do

    python ${bpe_dir}/learn_bpe.py -s 10000 < ${NAME}.txt > ${NAME}.BPE
    python ${bpe_dir}/apply_bpe.py -c ${NAME}.BPE < ${NAME}.txt > ${NAME}_BPE.txt

done
