python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IA.txt > simulations/IA.BPE
python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IB.txt > simulations/IB.BPE
python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IIA.txt > simulations/IIA.BPE
python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IIB.txt > simulations/IIB.BPE

python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IA_F.txt > simulations/IA_F.BPE
python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IB_F.txt > simulations/IB_F.BPE
python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IIA_F.txt > simulations/IIA_F.BPE
python ../subword-nmt-master/learn_bpe.py -s 10000 < simulations/IIB_F.txt > simulations/IIB_F.BPE

python ../subword-nmt-master/apply_bpe.py -c simulations/IA.BPE < simulations/IA.txt > simulations/IA_BPE.txt
python ../subword-nmt-master/apply_bpe.py -c simulations/IB.BPE < simulations/IB.txt > simulations/IB_BPE.txt
python ../subword-nmt-master/apply_bpe.py -c simulations/IIA.BPE < simulations/IIA.txt > simulations/IIA_BPE.txt
python ../subword-nmt-master/apply_bpe.py -c simulations/IIB.BPE < simulations/IIB.txt > simulations/IIB_BPE.txt

python ../subword-nmt-master/apply_bpe.py -c simulations/IA_F.BPE < simulations/IA_F.txt > simulations/IA_F_BPE.txt
python ../subword-nmt-master/apply_bpe.py -c simulations/IB_F.BPE < simulations/IB_F.txt > simulations/IB_F_BPE.txt
python ../subword-nmt-master/apply_bpe.py -c simulations/IIA_F.BPE < simulations/IIA_F.txt > simulations/IIA_F_BPE.txt
python ../subword-nmt-master/apply_bpe.py -c simulations/IIB_F.BPE < simulations/IIB_F.txt > simulations/IIB_F_BPE.txt
