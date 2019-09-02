mkdir data
mv train_preprocessed.txt data && mv train_preprocessed_shortest.txt data && mv test_preprocessed.txt data && mv test_preprocessed_shortest.txt data
cd subword-nmt-master && python3 learn_bpe.py -s 10000 < ../data/train_preprocessed.txt > train_preprocessed.code && python3 learn_bpe.py -s 10000 < ../data/train_preprocessed_shortest.txt > train_preprocessed_shortest.code
python3 apply_bpe.py -c train_preprocessed.code < ../data/train_preprocessed.txt > ../data/train_preprocessed_BPE.txt && python3 apply_bpe.py -c train_preprocessed.code < ../data/test_preprocessed.txt > ../data/test_preprocessed_BPE.txt && python3 apply_bpe.py -c train_preprocessed_shortest.code < ../data/train_preprocessed_shortest.txt > ../data/train_preprocessed_shortest_BPE.txt && python3 apply_bpe.py -c train_preprocessed_shortest.code < ../data/test_preprocessed_shortest.txt > ../data/test_preprocessed_shortest_BPE.txt
cd .. && mv glove.sh glove && cp ./data/train_preprocessed.txt glove && cp ./data/train_preprocessed_BPE.txt glove && cp ./data/train_preprocessed_shortest.txt glove && cp ./data/train_preprocessed_shortest_BPE.txt glove && cd glove && bash glove.sh
mv vectors_train_preprocessed.txt ../data && mv vectors_train_preprocessed_BPE.txt ../data && mv vectors_train_preprocessed_shortest.txt ../data && mv vectors_train_preprocessed_shortest_BPE.txt ../data




