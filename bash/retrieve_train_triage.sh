#!/bin/bash
filename_train=train_pmids_triage.txt
filelines_train=`cat $filename_train`

echo Start
mkdir -p ../train_triage/pubtator
for line in $filelines_train ; do
    perl PreProcessing.pl -t PMID -i $line -o ../train_triage/pubtator/$line.txt
done
