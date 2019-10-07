#!/bin/bash
DATA=("triage_data_preprocessed_train" "triage_data_original_train")

for NAME in ${DATA[@]}; do

    python learn_bpe.py -s 10000 < ${NAME}.txt > ${NAME}.BPE
    python apply_bpe.py -c ${NAME}.BPE < ${NAME}.txt > ${NAME}_BPE.txt

done
