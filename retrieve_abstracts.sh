#!/bin/bash
filename='list_PMID.txt'
filelines=`cat $filename`
echo Start
for line in $filelines ; do
    perl PreProcessing.pl -t PMID -i $line -o articles_pubtator/$line.txt
done

filename='list_PMID_test.txt'
filelines=`cat $filename`
echo Start
for line in $filelines ; do
    perl PreProcessing.pl -t PMID -i $line -o articles_pubtator_test/$line.txt
done
