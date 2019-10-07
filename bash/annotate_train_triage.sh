#!/bin/bash
INPUT=../train_triage/pubtator
mkdir -p ../train_triage/GNormPlus
mkdir -p ../train_triage/text
OUTPUT=../train_triage/GNormPlus
SETUP=setup.txt

java -Xmx20G -Xms20G  -jar GNormPlus.jar $INPUT $OUTPUT $SETUP
