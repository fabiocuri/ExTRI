python subword-nmt-master/learn_bpe.py -s 10000 < TRI/simulations/simulation_IA_preprocessed.txt > TRI/simulations/simulation_IA_preprocessed.BPE
python subword-nmt-master/learn_bpe.py -s 10000 < TRI/simulations/simulation_IB_preprocessed.txt > TRI/simulations/simulation_IB_preprocessed.BPE
python subword-nmt-master/learn_bpe.py -s 10000 < TRI/simulations/simulation_IIA_preprocessed.txt > TRI/simulations/simulation_IIA_preprocessed.BPE
python subword-nmt-master/learn_bpe.py -s 10000 < TRI/simulations/simulation_IIB_preprocessed.txt > TRI/simulations/simulation_IIB_preprocessed.BPE

python subword-nmt-master/apply_bpe.py -c TRI/simulations/simulation_IA_preprocessed.BPE < TRI/simulations/simulation_IA_preprocessed.txt > TRI/simulations/simulation_IA_preprocessed_BPE.txt
python subword-nmt-master/apply_bpe.py -c TRI/simulations/simulation_IB_preprocessed.BPE < TRI/simulations/simulation_IB_preprocessed.txt > TRI/simulations/simulation_IB_preprocessed_BPE.txt
python subword-nmt-master/apply_bpe.py -c TRI/simulations/simulation_IIA_preprocessed.BPE < TRI/simulations/simulation_IIA_preprocessed.txt > TRI/simulations/simulation_IIA_preprocessed_BPE.txt
python subword-nmt-master/apply_bpe.py -c TRI/simulations/simulation_IIB_preprocessed.BPE < TRI/simulations/simulation_IIB_preprocessed.txt > TRI/simulations/simulation_IIB_preprocessed_BPE.txt
