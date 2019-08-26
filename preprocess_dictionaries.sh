echo "Converting hyphens to underscores and removing punctuations"
sed 's/-/ /g' ./dictionaries/nodbtf_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/nodbtf_official_underscore.dic.txt2
sed 's/-/ /g' ./dictionaries/dbtf_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/dbtf_syn_underscore.dic.txt2
sed 's/-/ /g' ./dictionaries/nodbtf_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/nodbtf_syn_underscore.dic.txt2
sed 's/-/ /g' ./dictionaries/dbtf_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/dbtf_official_underscore.dic.txt2
sed 's/-/ /g' ./dictionaries/nodbtf_long_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/nodbtf_long_syn_underscore.dic.txt2
sed 's/-/ /g' ./dictionaries/nodbtf_long_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/nodbtf_long_official_underscore.dic.txt2
sed 's/-/ /g' ./dictionaries/dbtf_long_syn_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/dbtf_long_syn_underscore.dic.txt2
sed 's/-/ /g' ./dictionaries/dbtf_long_official_underscore.dic | sed 's/_/ /g' | sed 's/\./ /g' | sed 's/\// /g' | sed 's/)/ /g' | sed 's/(/ /g' > ./dictionaries/dbtf_long_official_underscore.dic.txt2
