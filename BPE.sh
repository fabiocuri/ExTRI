DATA=("IA" "IB" "IIA" "IIB" "IA_F" "IB_F" "IIA_F" "IIB_F")

for NAME in ${DATA[@]}; do

    python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/${NAME}.txt > simulations/${NAME}.BPE
    python ../subword-nmt-master/apply_bpe.py -c simulations/${NAME}.BPE < simulations/${NAME}.txt > simulations/${NAME}_BPE.txt
