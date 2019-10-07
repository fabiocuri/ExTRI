#!/bin/bash
for a in ../train_triage/text/*.txt
do
sed 's/-/ /g' $a | sed 's/_/ /g' | sed 's/,/ /g' | sed 's/:/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > $a.txt2
done
wait
cd .. && cd scripts && python3 NTNU_NER.py --folder train_triage
wait
for x in ../train_triage/text/*.out
do
sort -u -k3,4 $x | sed 's/#/\n#/g' | sed 's/Notes-/Notes /g' | awk '{if($2!="AnnotatorNotes") print $1"\t"$2" "$3" "$4"\t"$5" "$6" "$7" "$8" "$9" "$10" "$11" "$12; else print $1"\t"$2" "$3"\t"$4" "$5" "$6" "$7" "$8}' > $x.minfner
done
mkdir -p ../train_triage/NTNU
mkdir -p ../train_triage/merged
for i in ../train_triage/text/*.minfner; do mv "$i" ../train_triage/NTNU; done
