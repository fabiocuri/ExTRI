#!/usr/bin/env python
# coding: utf-8

from keras.layers import Dropout, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Embedding
from ExportPMIDs import read_as_list
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, RMSprop
import spacy
import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from imblearn.over_sampling import ADASYN
from keras.utils.vis_utils import plot_model
import itertools
import argparse
from PreprocessExportTrainingData import categorize_features

def categorize_list(l, dictionary):

    l_categorized = []
    for l_ in l:
        l_categorized.append(dictionary[l_])

    return l_categorized

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

class Attention(Layer):
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

def run_RNN(X, labels, path_to_glove, MNW, LSTMD, ATT, OPT, model_name):

    print('Running ' + str(model_name))

    # Hyper-parameters of the model

    MAX_SEQUENCE_LENGTH = MSL
    MAX_NB_WORDS = MNW
    LSTM_DIM = LSTMD
    ATTENTION = ATT
    OPTIMIZER = OPT

    # Categorize target labels: it supposes labels is originally boolean!

    d = {'false': 0, 'true': 1, 'nan': 2}
    labels = [d[x] for x in labels]
    n_labels = len(set(labels))
    labels = to_categorical(np.asarray(labels))

    texts = []

    for idx in X:
        text = BeautifulSoup(idx, 'html.parser')
        texts.append(str(text.get_text().encode()))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Shuffle data

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    # Run Model with Cross-validation

    n_folds = 10
    l_precision, l_recall, l_f1, l_val_accuracy = [], [], [], []

    for n_ in range(n_folds):

        print('Running fold ' + str(n_))

        # Split data

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]

        # Oversampling of minority label in training set

        ada = ADASYN(random_state=42, sampling_strategy='minority')
        X_train_resampled, y_train_resampled = ada.fit_sample(x_train, y_train)
        y_train_resampled = to_categorical(np.asarray(y_train_resampled))

        # Build GloVe dictionaries

        embeddings_index = {}
        f = open(cwd + '/glove/' + path_to_glove, encoding='utf8')
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

        if ATTENTION:

            model_type = "Bidirectional LSTM with Attention"

            out = Bidirectional(LSTM(LSTM_DIM, return_sequences=True, dropout=0.30, recurrent_dropout=0.30))(embedded_sequences)
            out = Attention(MAX_SEQUENCE_LENGTH)(out)
            out = Dense(LSTM_DIM, activation="relu")(out)

        else:

            model_type = "Bidirectional LSTM" 

            out = Bidirectional(LSTM(LSTM_DIM))(embedded_sequences)

        out = Dropout(0.30)(out)
        out = Dense(n_labels, activation="softmax")(out)
        model = Model(sequence_input, out) 

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

    f = open(cwd + '/results_RNN.txt', "a")
    f.write(l_results + "\n")
    f.close()

if '__main__' == __name__:

    encoding = 'latin-1'

    cwd = os.getcwd()
    nlp = spacy.load('en_core_web_sm')
    plt.switch_backend('agg')

    parser = argparse.ArgumentParser(description='Hyper-parameters of the model.')
    parser.add_argument('--data', type=str, help="""Preprocessed .txt file with text.""")
    parser.add_argument('--labels', type=str, help="""Preprocessed .txt file with labels.""")
    parser.add_argument('--max_num_words', type=int, help="""Maximum number of words.""")
    parser.add_argument('--dim_LSTM', type=int, help="""Dimension of the LSTM layer.""")
    parser.add_argument('--attention', type=str, default=None, help="""Whether to use the attention mechanism.""")
    parser.add_argument('--optimizer', type=str, default=None, help="""Optimizer of the RNN.""")

    args = parser.parse_args()

    X = args.data
    y = args.labels
    MNW = args.max_num_words
    LSTMD = args.dim_LSTM
    ATT = args.attention
    OPT = args.optimizer

    MSL = 100 # Maximum sequence lengths.
    VALIDATION_SPLIT = 0.2 # Validation % 

    train = read_as_list(cwd + '/simulations/' + X + '.txt', encoding=encoding)
    labels = read_as_list(cwd + '/simulations/' + y + '.txt', encoding=encoding)
    path_to_glove = 'vectors_' + X + '.txt'

    run_RNN(train, labels, path_to_glove, MNW, LSTMD, ATT, OPT, "RNN_%s_%s_%s_%s_%s" % (str(X), str(MNW), str(LSTMD), str(ATT), str(OPT)))
