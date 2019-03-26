#!/bin/bash
filename_train=../Triage/PMID_train.txt
filename_test=../Triage/PMID_test.txt
filelines_train=`cat $filename_train`
filelines_test=`cat $filename_test`
output_train=train_abstracts
output_test=test_abstracts

echo Start
for line in $filelines_train ; do
    perl PreProcessing.pl -t PMID -i $line -o ../Triage/pubtator/${output_train}/$line.txt
done

echo Start
for line in $filelines_test ; do
    perl PreProcessing.pl -t PMID -i $line -o ../Triage/pubtator/${output_test}/$line.txt
done
