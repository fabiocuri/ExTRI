DATA=("IB_F" "IIB_F" "IB_F_BPE" "IIB_F_BPE")
MAX_NUM_WORDS=("500" "20000")
DIM_LSTM=("100" "500")
ATTENTION=("None" "Att")
OPTIMIZER=("adam" "rmsprop")
OVERSAMPLING=("SMOTE" "ROS" "ADASYN")
MODEL_SELECTION="yes"

for i in ${DATA[@]}; do
    for j in ${MAX_NUM_WORDS[@]}; do
        for k in ${DIM_LSTM[@]}; do
            for l in ${ATTENTION[@]}; do
                for o in ${OPTIMIZER[@]}; do

                    python3 RunRNN.py --data ${i} --labels labels --max_num_words ${j} --dim_LSTM ${k} --attention ${l} --optimizer ${o} --oversampling ${OVERSAMPLING} --model_selection ${MODEL_SELECTION} 

                done
            done
        done
    done
done
