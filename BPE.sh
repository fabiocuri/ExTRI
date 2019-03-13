DATA=("IA" "IB" "IIA" "IIB")

for NAME in ${DATA[@]}; do

    python subword-nmt-master/learn_bpe.py -s 10000 < simulations/${NAME}.txt > simulations/${NAME}.BPE
    python subword-nmt-master/learn_bpe.py -s 10000 < simulations/${NAME}_F.txt > simulations/${NAME}_F.BPE

    python subword-nmt-master/apply_bpe.py -c simulations/${NAME}.BPE < simulations/${NAME}.txt > simulations/${NAME}_BPE.txt
    python subword-nmt-master/apply_bpe.py -c simulations/${NAME}_F.BPE < simulations/${NAME}_F.txt > simulations/${NAME}_F_BPE.txt

    python subword-nmt-master/apply_bpe.py -c simulations/${NAME}.BPE < simulations/${NAME}_test.txt > simulations/${NAME}_test_BPE.txt
    python subword-nmt-master/apply_bpe.py -c simulations/${NAME}_F.BPE < simulations/${NAME}_test_F.txt > simulations/${NAME}_test_F_BPE.txt

done
