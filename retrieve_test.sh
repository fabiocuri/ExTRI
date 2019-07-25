#!/bin/bash
filename_train=test.txt
filelines_train=`cat $filename_train`

echo Start
mkdir -p ./test/pubtator
for line in $filelines_train ; do
    perl PreProcessing.pl -t PMID -i $line -o ./test/pubtator/$line.txt
done
