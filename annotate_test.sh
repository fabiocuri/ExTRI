#!/bin/bash
INPUT=./test/pubtator
mkdir -p ./test/GNormPlus
mkdir -p ./test/text
OUTPUT=./test/GNormPlus
SETUP=setup.txt

java -Xmx20G -Xms20G  -jar GNormPlus.jar $INPUT $OUTPUT $SETUP
