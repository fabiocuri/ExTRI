#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao
E-mail: fcuri91@gmail.com
Date: 07.10.2019
'''

import os
import keras
import argparse
import numpy as np
from numpy import array
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
plt.switch_backend('agg')
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import Adam, RMSprop
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from export_abstracts import read_as_list, write_list
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.layers import Dense, Input, Flatten, Embedding, Dropout, LSTM, Bidirectional

class Attention(Layer):

    ''' Attention layer '''

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

def run_RNN(X, test, labels, path_to_glove, REPORT, out_folder):

    ''' Recurrent Neural Networks with Attention enabled '''

    # Hyper-parameters of the model

    MAX_SEQUENCE_LENGTH = 100
    MAX_NB_WORDS = 500
    LSTM_DIM = 100

    # Categorize target labels !

    n_labels = len(set(labels))

    values = array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    labels = onehot_encoder.fit_transform(integer_encoded)

    texts, texts_test = [], []

    for idx in X:
        text = BeautifulSoup(idx, 'html.parser')
        texts.append(str(text.get_text().encode()))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    if REPORT == 'yes':
        for idx in test:
            text = BeautifulSoup(idx, 'html.parser')
            texts_test.append(str(text.get_text().encode()))

        tokenizer_test = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer_test.fit_on_texts(texts)
        sequences_test = tokenizer_test.texts_to_sequences(texts_test)
        word_index_test = tokenizer_test.word_index
        data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    # Run Model with Cross-validation

    if REPORT == 'yes':
        n_folds = 1
    if REPORT == 'no':
        n_folds = 10

    VALIDATION_SPLIT = 0.2
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

        if REPORT == 'yes':
            x_train = data
            y_train = labels
            x_val = data
            y_val = labels

        if REPORT == 'no':
            x_train = data[:-nb_validation_samples]
            y_train = labels[:-nb_validation_samples]
            x_val = data[-nb_validation_samples:]
            y_val = labels[-nb_validation_samples:]

        # Oversampling of minority label in training set

        ada = RandomOverSampler(random_state=42, sampling_strategy='minority')
        X_train_resampled, y_train_resampled = ada.fit_sample(x_train, y_train)
        y_train_resampled = onehot_encoder.fit_transform(y_train_resampled)

        # Build GloVe dictionaries

        embeddings_index = {}
        f = open(path_to_glove, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.random.random((len(word_index) + 1, MAX_SEQUENCE_LENGTH))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    MAX_SEQUENCE_LENGTH,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)


        model_type = out_folder + "/Bidirectional LSTM with Attention"

        out = Bidirectional(LSTM(LSTM_DIM, return_sequences=True, dropout=0.30, recurrent_dropout=0.30))(embedded_sequences)
        out = Attention(MAX_SEQUENCE_LENGTH)(out)
        out = Dense(LSTM_DIM, activation="relu")(out)

        out = Dropout(0.30)(out)
        out = Dense(n_labels, activation="softmax")(out)
        model = Model(sequence_input, out)

        # Save model architecture

        if not os.path.isfile(model_type + '.png'):

            plot_model(model, to_file=model_type+'.png', show_shapes=True, show_layer_names=True)

        # Model optimizer and metrics

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Model parameters

        if not os.path.exists(out_folder + '/models'):
            os.makedirs(out_folder + '/models')

        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 0, mode= 'min')
        model_filepath_weights = out_folder + '/models/triage_' + path_to_glove.split('/')[2].split('.')[0] + '_' + str(n_) + '.h5'
        model_filepath_json = out_folder + '/models/triage_' + path_to_glove.split('/')[2].split('.')[0] + '_' + str(n_) + '.json'

        checkpoint = ModelCheckpoint(model_filepath_weights, monitor='val_acc', verbose = 0, save_best_only=True, mode='max')
        callbacks_list = [early_stopping, checkpoint]

        history = model.fit(X_train_resampled, y_train_resampled, validation_data=(x_val, y_val), epochs=100, callbacks=callbacks_list, verbose = 0)

        # Predictions

        if REPORT == 'yes':
            y_pred = model.predict(data_test)
            y_pred = y_pred.argmax(axis=-1)
            y_pred = [str(x) for x in y_pred]

            write_list(y_pred, out_folder + '/models/triage_predictions.txt', iterate=True, encoding=encoding)

            # serialize model to JSON
            model_json = model.to_json()
            with open(model_filepath_json, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            keras.models.save_model(model, model_filepath_weights)
            print("Saved model to disk")

        if REPORT == 'no':
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

    if REPORT == 'no':
        l_results = 'triage_RNN' + '\t' + str(np.mean(l_precision)) + '\t' + str(np.mean(l_recall)) + '\t' + str(np.mean(l_f1))
        print(l_results)

if '__main__' == __name__:

    ''' Run Recurrent Neural Network for triage. '''

    encoding = 'latin-1'

    parser = argparse.ArgumentParser(description='Hyper-parameters of the model.')
    parser.add_argument('--train', type=str, help="""Train file.""")
    parser.add_argument('--test', type=str, help="""Test file.""")
    parser.add_argument('--report', type=str, help="""If yes, predict unseen data.""")
    parser.add_argument('--labels', type=str, help="""Labels file.""")
    parser.add_argument('--glove', type=str, help="""Glove directory.""")
    parser.add_argument('--o', type=str, help="""Output directory.""")

    args = parser.parse_args()

    train = read_as_list(args.train, encoding='latin-1')
    if args.report == 'yes':
        test = read_as_list(args.test, encoding='latin-1')
    if args.report == 'no':
        test = []
    labels = read_as_list(args.labels, encoding='latin-1')
    path_to_glove = args.glove + '/vectors_' + args.train.split('/')[-1]

    run_RNN(train, test, labels, path_to_glove, args.report, args.o)
