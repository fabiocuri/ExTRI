#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import stopwords 
import string
import os
import pandas as pd
import numpy as np
import nltk
from collections import defaultdict
from langdetect import detect
from textblob import TextBlob

nltk.download('punkt')

stop_words = set(stopwords.words('english')) 
punctuation = list(string.punctuation)

cwd = os.getcwd()

# Import training data

training_data = pd.DataFrame.from_csv(cwd + '/abstracts.all.labeled.csv', sep='\t|\n')
PMIDs = list(training_data.index)

# Import annotated abstracts

abstracts_annotated = defaultdict(list)

for root, dirs, files in os.walk(cwd + '/annotated_training_data'):
    for file in files:
        if file.endswith('.txt'):
            PMID=file.replace('.txt', '')
            with open(cwd + '/annotated_training_data/' + file, "r") as f:
                if PMID in files:
                    continue
                abstracts_annotated[str(PMID)] = f.readlines()


def replace_entities(df, annotated_dictionary):
    
    # Replace abstracts by extracted entitites.
    
    abstracts_with_entities, titles_with_entities = [], []

    for PMID in PMIDs:

        abstract = df['abstract'][PMID]
        title = df['title'][PMID]
        
        entities = annotated_dictionary[str(PMID)]

        if len(entities) > 3: # If there are entitites extracted from GNormPLUS ...

            if abstract is not np.nan: # We assume titles are never np.nan ...

                abstract = nltk.word_tokenize(abstract)
                title = nltk.word_tokenize(title)

                entities = entities[2:] # Only entities

                for ent in entities:

                    ent_split = ent.split('\t')

                    if len(ent_split) > 3: # If not \n ...

                        key = ent_split[3]
                        entity = ent_split[4]

                        for i, w in enumerate(abstract):

                            if w == key:

                                abstract[i] = abstract[i].replace(w, entity.upper())

                        for i, w in enumerate(title):

                            if w == key:

                                title[i] = title[i].replace(w, entity.upper())

                abstracts_with_entities.append(' '.join(abstract))
                titles_with_entities.append(' '.join(title))

            else:
                
                abstracts_with_entities.append(np.nan)
                title = nltk.word_tokenize(title)
                titles_with_entities.append(' '.join(title))

        else:

            if abstract is not np.nan:
            
                abstract = nltk.word_tokenize(abstract)
                abstracts_with_entities.append(' '.join(abstract))

            else:

                abstracts_with_entities.append(np.nan)

            title = nltk.word_tokenize(title)
            titles_with_entities.append(' '.join(title))

    df['title_annotated'] = titles_with_entities        
    df['abstract_annotated'] = abstracts_with_entities
    
    return df

def dataframe_in_english(df):
    
    # Language identification to make sure all papers are in English
    
    lang_abstract = []
    
    for abstract in df['abstract']:

        if abstract is not np.nan:
            lang_abstract.append(detect(abstract))
        else:
            lang_abstract.append(np.nan)

    df['lang_abstract'] = lang_abstract

    list_languages = list(set(df['lang_abstract']))
    foreign_languages = [x for x in list_languages if x not in [np.nan, 'en']]

    for FL in foreign_languages:

        new_df = df[df['lang_abstract'] == FL]

        for abstract, abstract_annotated, title, title_annotated, PMID in zip(new_df['abstract'], new_df['abstract_annotated'], new_df['title'], new_df['title_annotated'], new_df.index):

            abstract_textblog = TextBlob(abstract)
            abstract_annotated_textblob = TextBlob(abstract_annotated) 
            
            title_textblog = TextBlob(title)
            title_annotated_textblob = TextBlob(title_annotated) 

            translated_abstract = str(abstract_textblog.translate(to="en"))
            translated_abstract_annotated = str(abstract_annotated_textblob.translate(to="en")) 
            
            translated_title = str(title_textblog.translate(to="en"))
            translated_title_annotated = str(title_annotated_textblob.translate(to="en")) 

            df.abstract[PMID] = translated_abstract
            df.abstract_annotated[PMID] = translated_abstract_annotated
            
            df.title[PMID] = translated_title
            df.title_annotated[PMID] = translated_title_annotated
    
    return df

def list_as_txt(l, l_name):
    
    with open(l_name, 'w') as f:
        for item in l:
            f.write("%s\n" % item)


training_data = replace_entities(training_data, abstracts_annotated)
training_data = dataframe_in_english(training_data)

abstracts = list(training_data['abstract'])
abstracts_annotated = list(training_data['abstract_annotated'])
titles = list(training_data['title'])
titles_annotated = list(training_data['title_annotated'])
has_tf = list(training_data['has_tf'])
mammal = list(training_data['mammal']) 

# Export training data

list_as_txt(abstracts, 'abstracts.txt')
list_as_txt(abstracts_annotated, 'abstracts_annotated.txt')
list_as_txt(titles, 'titles.txt')
list_as_txt(titles_annotated, 'titles_annotated.txt')
list_as_txt(has_tf, 'has_tf.txt')
list_as_txt(mammal, 'mammal.txt')
