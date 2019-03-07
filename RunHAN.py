#!/usr/bin/env python
# coding: utf-8

import os
os.environ['KERAS_BACKEND']='tensorflow'
import spacy
import argparse
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from imblearn.over_sampling import ADASYN
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Dense, Input, Flatten, Embedding, Dropout, LSTM, Bidirectional, GRU, TimeDistributed
from ExportPMIDs import read_as_list
from RunRNN import Attention

def run_HAN(X, labels, path_to_glove, MNW, LSTMD, OPT, model_name):

    ''' Hierarchical Attention Network '''

    print('Running ' + str(model_name))

    # Hyper-parameters of the model

    MAX_SENT_LENGTH = MSL
    MAX_NB_WORDS = MNW
    LSTM_DIM = LSTMD
    OPTIMIZER = OPT

    # Categorize target labels: it supposes labels is originally boolean!

    d = {'false': 0, 'true': 1, 'nan': 2}
    labels = [d[x] for x in labels]
    n_labels = len(set(labels))
    labels = to_categorical(np.asarray(labels))

    texts, reviews = [], []

    for idx in X:
        text = BeautifulSoup(idx, 'html.parser')
        text = str(text.get_text().encode())
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j< MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                        data[i,j,k] = tokenizer.word_index[word]
                        k=k+1

    word_index = tokenizer.word_index

    # Run Model with Cross-validation

    n_folds = 10
    l_precision, l_recall, l_f1, l_val_accuracy = [], [], [], []

    for n_ in range(n_folds):

        print('Running fold ' + str(n_))

        # Shuffle data

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        # Split data

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]

        # Get shapes

        nsamples_xtrain, nx_xtrain, ny_xtrain = x_train.shape
        x_train = x_train.reshape((nsamples_xtrain,nx_xtrain*ny_xtrain))

        # Oversampling of minority label in training set

        ada = ADASYN(random_state=42, sampling_strategy='minority')
        X_train_resampled, y_train_resampled = ada.fit_sample(x_train, y_train)
        y_train_resampled = to_categorical(np.asarray(y_train_resampled))

        nsamples_xtrain_resampled, _ = X_train_resampled.shape
        X_train_resampled = X_train_resampled.reshape((nsamples_xtrain_resampled, nx_xtrain, ny_xtrain))

        # Build GloVe dictionaries

        embeddings_index = {}
        f = open(cwd + '/glove/' + path_to_glove, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.random.random((len(word_index) + 1, MAX_SENT_LENGTH))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    MAX_SENT_LENGTH,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=True)

        sequence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        model_type = "Hierarchical Attention Network"

        out = Bidirectional(LSTM(LSTM_DIM, return_sequences=True, dropout=0.30, recurrent_dropout=0.30))(embedded_sequences)
        out = TimeDistributed(Dense(2*LSTM_DIM))(out)
        out = Attention(MAX_SENT_LENGTH)(out)
        sentEncoder = Model(sequence_input, out)

        review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        out = Bidirectional(GRU(LSTM_DIM, return_sequences=True, dropout=0.30, recurrent_dropout=0.30))(review_encoder)
        out = TimeDistributed(Dense(2*LSTM_DIM))(out)
        out = Attention(MAX_SENTS)(out)
        out = Dropout(0.30)(out)
        out = Dense(n_labels, activation="softmax")(out)
        model = Model(review_input, out)

        # Save model architecture

        if not os.path.isfile(model_type+'.png'):

            plot_model(model, to_file=model_type+'.png', show_shapes=True, show_layer_names=True)

        # Model optimizer and metrics

        if OPTIMIZER=='adam':

            opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        if OPTIMIZER=='rmsprop':

            opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Model parameters

        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 0, mode= 'min')
        model_filepath = cwd + '/models/' + model_name + '_' + str(n_) + '.hdf5'

        checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose = 0, save_best_only=True, mode='min')
        callbacks_list = [early_stopping, checkpoint]

        if experiment_mode:
            callbacks_list = [early_stopping]

        history = model.fit(X_train_resampled, y_train_resampled, validation_data=(x_val, y_val), epochs=20, callbacks=callbacks_list, verbose = 0)

        # Predictions 

        y_pred = model.predict(x_val)
        y_pred = y_pred.argmax(axis=-1)
        y_val = y_val.argmax(axis=-1)
        y_pred = [str(x) for x in y_pred]
        y_val = [str(x) for x in y_val]

        # Results

        l_precision.append(precision_score(y_true=y_val, y_pred=y_pred, average='macro'))
        l_recall.append(recall_score(y_true=y_val, y_pred=y_pred, average='macro'))
        l_f1.append(f1_score(y_true=y_val, y_pred=y_pred, average='macro'))
        l_val_accuracy.append(history.history['val_acc'][-1])

    l_results = model_name + ' ' + str(np.mean(l_precision)) + ' ' + str(np.mean(l_recall)) + ' ' + str(np.mean(l_f1)) + ' ' + str(np.mean(l_val_accuracy))

    # Append results

    f = open(cwd + '/results_HAN.txt', "a")
    f.write(l_results + "\n")
    f.close()

if '__main__' == __name__:

    experiment_mode = True

    encoding = 'latin-1'

    cwd = os.getcwd()
    nlp = spacy.load('en_core_web_sm')
    plt.switch_backend('agg')

    parser = argparse.ArgumentParser(description='Hyper-parameters of the model.')
    parser.add_argument('--data', type=str, help="""Preprocessed .txt file with text.""")
    parser.add_argument('--labels', type=str, help="""Preprocessed .txt file with labels.""")
    parser.add_argument('--max_num_words', type=int, help="""Maximum number of words.""")
    parser.add_argument('--dim_LSTM', type=int, help="""Dimension of the LSTM layer.""")
    parser.add_argument('--optimizer', type=str, default=None, help="""Optimizer of the RNN.""")

    args = parser.parse_args()

    X = args.data
    y = args.labels
    MNW = args.max_num_words
    LSTMD = args.dim_LSTM
    OPT = args.optimizer

    MSL = 100 # Maximum sequence lengths.
    VALIDATION_SPLIT = 0.2 # Validation % 
    MAX_SENTS = 15

    train = read_as_list(cwd + '/simulations/' + X + '.txt', encoding=encoding)
    labels = read_as_list(cwd + '/simulations/' + y + '.txt', encoding=encoding)
    path_to_glove = 'vectors_' + X + '.txt'

    run_HAN(train, labels, path_to_glove, MNW, LSTMD, OPT, "HAN_%s_%s_%s_%s" % (str(X), str(MNW), str(LSTMD), str(OPT)))
