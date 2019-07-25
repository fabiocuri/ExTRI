#!/bin/bash
INPUT=../RE/test/pubtator
OUTPUT=../RE/test/GNormPlus
SETUP=setup.txt

java -Xmx20G -Xms20G  -jar GNormPlus.jar $INPUT $OUTPUT $SETUP
