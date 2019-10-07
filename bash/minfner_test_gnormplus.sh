#!/bin/bash
for a in ../test/GNormPlus/*.txt
do
sed 's/-/ /g' $a | sed 's/_/ /g' | sed 's/,/ /g' | sed 's/:/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > $a.txt2
done
wait
cd .. && cd scripts && python3 gnormplus_to_brat.py --folder test
wait
for x in ../test/GNormPlus/*.out
do
sort -u -k3,4 $x | sed 's/#/\n#/g' | sed 's/Notes-/Notes /g' | awk '{if($2!="AnnotatorNotes") print $1"\t"$2" "$3" "$4"\t"$5" "$6" "$7" "$8" "$9" "$10" "$11" "$12; else print $1"\t"$2" "$3"\t"$4" "$5" "$6" "$7" "$8}' > $x.minfner
done
