import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from export_abstracts import read_as_list, write_list

def preprocess(text):

    text = text.lower() # Lowercase
    text = text.strip() # Remove white spaces
    
    stop_words = list(set(list(stopwords.words('english')) + list(ENGLISH_STOP_WORDS) + list(STOP_WORDS))) # Remove stopwords
    stop_words = list(set(stop_words))
    tokens = word_tokenize(text)
    text = [i for i in tokens if not i in stop_words]

    stemmer= PorterStemmer()

    text = [stemmer.stem(word) for word in text] # Stemming
    text = ' '.join(text)

    text = re.sub(r'[^\x00-\x7f]', r' ', text)  # ASCII

    return text

if '__main__' == __name__:

    train = pd.read_csv('./train_data.csv')
    test = pd.read_csv('./test/merged_data.csv')

    train_out = [preprocess(x) for x in list(train['all_sentences'])]
    test_out = [preprocess(x) for x in list(test['all_sentences'])]

    write_list(train_out, './train_preprocessed.txt', True, 'latin-1')
    write_list(test_out, './test_preprocessed.txt', True, 'latin-1')

    train_out = [preprocess(x) for x in list(train['all_sentences_shortest'])]
    test_out = [preprocess(x) for x in list(test['all_sentences_shortest'])]

    write_list(train_out, './train_preprocessed_shortest.txt', True, 'latin-1')
    write_list(test_out, './test_preprocessed_shortest.txt', True, 'latin-1')
