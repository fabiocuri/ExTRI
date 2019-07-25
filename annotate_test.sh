#!/bin/bash
INPUT=./test/pubtator
mkdir -p ./test/GNormPlus
OUTPUT=./test/GNormPlus
SETUP=setup.txt

java -Xmx20G -Xms20G  -jar GNormPlus.jar $INPUT $OUTPUT $SETUP
