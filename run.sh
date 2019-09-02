#OPTION 1: 

#Train SVM
python3 SVM.py
#Score SVM
python3 score.py --predictions SVM_predictions_preprocessed_TF-IDF
python3 score.py --predictions SVM_predictions_preprocessed_BPE_TF-IDF
python3 score.py --predictions SVM_predictions_preprocessed_shortest_TF-IDF
python3 score.py --predictions SVM_predictions_preprocessed_shortest_BPE_TF-IDF
python3 score.py --predictions SVM_predictions_preprocessed_BoW
python3 score.py --predictions SVM_predictions_preprocessed_BPE_BoW
python3 score.py --predictions SVM_predictions_preprocessed_shortest_BoW
python3 score.py --predictions SVM_predictions_preprocessed_shortest_BPE_BoW

#OPTION 2: 

#Train RNN
python3 RunRNN.py --data preprocessed --max_num_words 500 --dim_LSTM 100
python3 RunRNN.py --data preprocessed_BPE --max_num_words 500 --dim_LSTM 100
python3 RunRNN.py --data preprocessed_shortest --max_num_words 500 --dim_LSTM 100
python3 RunRNN.py --data preprocessed_shortest_BPE --max_num_words 500 --dim_LSTM 100
python3 RunRNN.py --data preprocessed --max_num_words 500 --dim_LSTM 500
python3 RunRNN.py --data preprocessed_BPE --max_num_words 500 --dim_LSTM 500
python3 RunRNN.py --data preprocessed_shortest --max_num_words 500 --dim_LSTM 500
python3 RunRNN.py --data preprocessed_shortest_BPE --max_num_words 500 --dim_LSTM 500
python3 RunRNN.py --data preprocessed --max_num_words 1000 --dim_LSTM 100
python3 RunRNN.py --data preprocessed_BPE --max_num_words 1000 --dim_LSTM 100
python3 RunRNN.py --data preprocessed_shortest --max_num_words 1000 --dim_LSTM 100
python3 RunRNN.py --data preprocessed_shortest_BPE --max_num_words 1000 --dim_LSTM 100
python3 RunRNN.py --data preprocessed --max_num_words 1000 --dim_LSTM 500
python3 RunRNN.py --data preprocessed_BPE --max_num_words 1000 --dim_LSTM 500
python3 RunRNN.py --data preprocessed_shortest --max_num_words 1000 --dim_LSTM 500
python3 RunRNN.py --data preprocessed_shortest_BPE --max_num_words 1000 --dim_LSTM 500
#Score RNN
python3 score.py --predictions RNN_500_100_preprocessed_predictions
python3 score.py --predictions RNN_500_100_preprocessed_BPE_predictions
python3 score.py --predictions RNN_500_100_preprocessed_shortest_predictions
python3 score.py --predictions RNN_500_100_preprocessed_shortest_BPE_predictions
python3 score.py --predictions RNN_500_500_preprocessed_predictions
python3 score.py --predictions RNN_500_500_preprocessed_BPE_predictions
python3 score.py --predictions RNN_500_500_preprocessed_shortest_predictions
python3 score.py --predictions RNN_500_500_preprocessed_shortest_BPE_predictions
python3 score.py --predictions RNN_1000_100_preprocessed_predictions
python3 score.py --predictions RNN_1000_100_preprocessed_BPE_predictions
python3 score.py --predictions RNN_1000_100_preprocessed_shortest_predictions
python3 score.py --predictions RNN_1000_100_preprocessed_shortest_BPE_predictions
python3 score.py --predictions RNN_1000_500_preprocessed_predictions
python3 score.py --predictions RNN_1000_500_preprocessed_BPE_predictions
python3 score.py --predictions RNN_1000_500_preprocessed_shortest_predictions
python3 score.py --predictions RNN_1000_500_preprocessed_shortest_BPE_predictions



