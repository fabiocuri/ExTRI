#!/bin/bash
filename_train=../RE/test.txt
filelines_train=`cat $filename_train`

echo Start
for line in $filelines_train ; do
    perl PreProcessing.pl -t PMID -i $line -o ../RE/test/pubtator/$line.txt
done
