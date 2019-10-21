#!/bin/bash
filename_train=triage_train_pmids.txt
filelines_train=`cat $filename_train`
output_folder=/home/bsclife018/Desktop/ExTRI-master/triage

echo Start
mkdir -p ${output_folder}/pubtator
for line in $filelines_train ; do
    perl PreProcessing.pl -t PMID -i $line -o ${output_folder}/pubtator/$line.txt
done
