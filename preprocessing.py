import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess(text):

    text = text.lower() # Lowercase
    text = re.sub(r'\d+', '', text) # Remove integers
    text = text.strip() # Remove white spaces
    
    stop_words = list(set(list(stopwords.words('english')) + list(ENGLISH_STOP_WORDS) + list(STOP_WORDS))) # Remove stopwords
    stop_words = list(set(stop_words))
    tokens = word_tokenize(text)
    text = [i for i in tokens if not i in stop_words]

    stemmer= PorterStemmer()

    text = [stemmer.stem(word) for word in text] # Stemming
    text = ' '.join(text)

    return text
