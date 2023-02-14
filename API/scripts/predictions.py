import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# download stop words
nltk.download('stopwords')
tags = pd.read_csv("scripts/vectorized_tags.csv")
tags = tags.columns.to_list()

def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('/', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

# Stop words
from nltk.corpus import stopwords
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')', '>', '<']\
                                               + ['like', 'use', 'using', 'want', 'way', 'strong', 'errors', 'error', 'pre', 'code']\
                                               + ['.+[1-9].+']

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)
from nltk.stem import WordNetLemmatizer

def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

    
# fonction de nettoyage
def cleaning(text):
    text = text.replace('<pre>(.|\n)+?</pre>', '', regex=True)
    text = text.replace('<a(.|\n)+?>', '', regex=True)
    text = text.replace('<code>(.|\n)+?</code>', '', regex=True)
    text = transform_bow_fct(text)
    return(text)

# fonction vectoriser tfidf
def vectoriser(text):
    filename = "./scripts/cv_model.sav"
    multi_tfidf_vectorizer = pickle.load(open(filename, 'rb'))
    return multi_tfidf_vectorizer.transform([text])

# fonction modeliser
def modeliser(x):

    filename = "./scripts/ovr_modelsav"
    logistic_regression = pickle.load(open(filename, 'rb'))
    y = logistic_regression.predict(x)
    print(y)
    return y

# retourne les étiquettes pour les predictions
def labels(y, labels):
    pred = []
    for i, is_label in enumerate(y[0]):
        if is_label == 0:
            pass
        else :
            pred.append(labels[i])
    return pred


def predire(text):
    a = cleaning(text)
    b = vectoriser(a)
    c = modeliser(b)
    predicted_tags = labels(c, labels=tags)
    return predicted_tags

