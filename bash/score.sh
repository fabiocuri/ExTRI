#!/bin/bash
#Triage: Run and score best model
cd ../models && rm -r * && cd ../scripts && python3 build_data_triage.py --folder test
python3 RNN_triage.py --train triage_data_preprocessed_train --test triage_data_preprocessed_test --report yes
python3 score.py --task triage --predictions triage_RNN_predictions

#RE: Run and score best model
python3 RF.py --report yes
python3 score.py --task re --predictions re_RF_predictions_preprocessed_BoW

#Remove files
cd .. && mv requirements.txt data && rm -r *.txt && mv data/requirements.txt ..
