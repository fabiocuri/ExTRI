echo "Converting hyphens to underscores and removing punctuations"
sed 's/-/ /g' nodbtf_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > nodbtf_official_underscore.dic.txt2
sed 's/-/ /g' dbtf_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > dbtf_syn_underscore.dic.txt2
sed 's/-/ /g' nodbtf_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > nodbtf_syn_underscore.dic.txt2
sed 's/-/ /g' dbtf_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > dbtf_official_underscore.dic.txt2
sed 's/-/ /g' nodbtf_long_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > nodbtf_long_syn_underscore.dic.txt2
sed 's/-/ /g' nodbtf_long_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > nodbtf_long_official_underscore.dic.txt2
sed 's/-/ /g' dbtf_long_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > dbtf_long_syn_underscore.dic.txt2
sed 's/-/ /g' dbtf_long_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > dbtf_long_official_underscore.dic.txt2
