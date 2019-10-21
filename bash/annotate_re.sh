#!/bin/bash
output_folder=/home/bsclife018/Desktop/ExTRI-master/re
INPUT=${output_folder}/pubtator
mkdir -p ${output_folder}/GNormPlus
mkdir -p ${output_folder}/text
OUTPUT=${output_folder}/GNormPlus
SETUP=setup.txt

java -Xmx20G -Xms20G  -jar GNormPlus.jar $INPUT $OUTPUT $SETUP
