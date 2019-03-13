DATA=("IB_F" "IIB_F" "IB_F_BPE" "IIB_F_BPE")
MAX_NUM_WORDS=("500" "20000")
DIM_CONV=("64" "128" "256")
OPTIMIZER=("adam" "rmsprop")
OVERSAMPLING=("SMOTE" "ROS" "ADASYN")
MODEL_SELECTION="yes"

for i in ${DATA[@]}; do
    for j in ${MAX_NUM_WORDS[@]}; do
        for k in ${DIM_CONV[@]}; do
            for o in ${OPTIMIZER[@]}; do

                    python3 RunCNN.py --data ${i} --labels labels --max_num_words ${j} --dim_CONV ${k} --optimizer ${o} --oversampling ${OVERSAMPLING} --model_selection ${MODEL_SELECTION} 

            done
        done
    done
done
