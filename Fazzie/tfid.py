# Developer：Fazzie
# Time: 2021/11/2519:12
# File name: tfid.py
# Development environment: Anaconda Python


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('./train_set.csv/train_set.csv', sep='\t', nrows=50000)

# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
# vectorizer = CountVectorizer()
# bagWord = vectorizer.fit_transform(corpus).toarray()
#
# print(bagWord)

# vectorizer = CountVectorizer(max_features=3000)
# train_test = vectorizer.fit_transform(train_df['text'])
# print(train_test)
clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000])
# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, smooth_idf=True)
train_test = tfidf.fit_transform(train_df['text'])
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
