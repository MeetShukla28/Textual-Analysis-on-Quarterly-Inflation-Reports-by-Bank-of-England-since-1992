# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 12:50:37 2021

@author: PC
"""

#Main
import PyPDF2
import re
import os

for foldername,subfolders,files in os.walk(r"C:/Meet College/Research (Professor Tatevik and Ryan)"):
    text_full = ' '
    for file in files:
        # open the pdf file
        object = PyPDF2.PdfFileReader(os.path.join(foldername,file))
        num_pages = object.numPages
        count = 0
        # while loop will read each page.
        while count < num_pages:
            pageObj = object.getPage(count)
            count += 1
            texts = pageObj.extractText()
            print('Page number:', count)
            print(texts)
            text_full = text_full + texts
            
            
            

#Tokenizing
import pandas as pd
import nltk
nltk.download('punkt')
pd.set_option('display.max_colwidth', None)
sentences = nltk.sent_tokenize(text_full)
print(sentences)
words = nltk.word_tokenize(text_full)
print(words)

#stemming
'''nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)'''
    
'''#Lemmatization
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)'''
    
'''#Bag of Words
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(text_full)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.Lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)'''
    
'''# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()'''

'''# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()'''

'''#Word2Vec
import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re
# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text_full)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.key_to_index

# Finding Word Vectors
vector = model.wv['inflation']

# Most similar words
similar = model.wv.most_similar('report')'''

# Word Embedding

import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
voc_size=10000
onehot_repr=[one_hot(words,voc_size)] 
print(onehot_repr)
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np
sent_length=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
dim=10
model=Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')
model.summary()
print(model.predict(embedded_docs))
embedded_docs[0]
print(model.predict(embedded_docs)[0])