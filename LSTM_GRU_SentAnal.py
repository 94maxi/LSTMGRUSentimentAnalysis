#!/usr/bin/env python
# coding: utf-8




import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.regularizers import Regularizer
from keras.layers import Dense, GRU, Flatten, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re










import pandas as pd
data = pd.read_csv('Reviews_LSTM.csv')
data.head()





data['text'] = data['text'].apply(lambda x: x.lower())
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
print(len(X))





embed_dim = 100


model = Sequential()
model.add(Embedding(max_words, embed_dim,input_length = X.shape[1] ))
# You can swap LSTM for GRU here
model.add(Bidirectional(LSTM(32, dropout=0.2,return_sequences=True)))
model.add(LSTM(16, dropout=0.2))
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='softmax',kernel_regularizer='l1'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())





Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.1, random_state = 1)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
print(Y_train,X_train)




from keras.callbacks import EarlyStopping
callback =EarlyStopping(monitor='val_loss', patience=3)





history=model.fit(X_train, Y_train, epochs = 10,batch_size=32, verbose = 2, callbacks=callback, validation_data=(X_val,Y_val))





scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





pred_text=["Sample Text To Predict"]
X_pred = tokenizer.texts_to_sequences(pred_text)
X_pred = pad_sequences(X_pred)
prediction=model.predict_classes(X_pred)
print("bad" if prediction ==0 else "good")

