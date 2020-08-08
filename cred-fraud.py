#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import tensorflow as tf

np.random.seed(2)




data = pd.read_csv('cred-data.csv')


# ## Data Exploration



data.head()


# ## Data Pre-Processing



data.drop('Time', axis=1, inplace=True)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



data['NormalisedAmount'] = scaler.fit_transform(data[['Amount']])
data.drop('Amount', axis=1, inplace=True)



X = data.drop('Class', axis=1)
y = data['Class']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)



X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ## Building the Nueral Network


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout




model = Sequential()

model.add(Dense(units=16, activation='relu', input_dim = 29))
model.add(Dense(units=24, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))




model.summary()


# ## Training the Network


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_history = model.fit(x=X_train, y=y_train, epochs=5, batch_size=15, verbose=1, validation_data=(X_test,y_test))


y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns


confusion_matrix(y_test, y_pred.round())





