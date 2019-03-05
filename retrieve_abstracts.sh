#!/bin/bash
filename='../ExTRI-master/list_pmid.txt'
filelines=`cat $filename`
echo Start
for line in $filelines ; do
    perl PreProcessing.pl -t PMID -i $line -o input/$line.txt
done
