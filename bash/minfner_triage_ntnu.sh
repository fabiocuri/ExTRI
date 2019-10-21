#!/bin/bash
triage_folder=triage
main_folder=/home/bsclife018/Desktop/ExTRI-master
script_dir=${main_folder}/scripts
output_folder=${main_folder}/${triage_folder}
dic_folder=${main_folder}/data/dictionaries
for a in ${output_folder}/text/*.txt
do
sed 's/-/ /g' $a | sed 's/_/ /g' | sed 's/,/ /g' | sed 's/:/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > $a.txt2
done
wait
python3 ${script_dir}/NTNU_NER.py --i1 ${output_folder}/text --i2 ${dic_folder}
wait
for x in ${output_folder}/text/*.out
do
sort -u -k3,4 $x | sed 's/#/\n#/g' | sed 's/Notes-/Notes /g' | awk '{if($2!="AnnotatorNotes") print $1"\t"$2" "$3" "$4"\t"$5" "$6" "$7" "$8" "$9" "$10" "$11" "$12; else print $1"\t"$2" "$3"\t"$4" "$5" "$6" "$7" "$8}' > $x.minfner
done
mkdir -p ${output_folder}/NTNU
mkdir -p ${output_folder}/merged
for i in ${output_folder}/text/*.minfner; do mv "$i" ${output_folder}/NTNU; done
