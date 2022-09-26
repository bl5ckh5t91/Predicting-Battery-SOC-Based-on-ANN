import sys
import datetime
from source import fileRead, preProcessing, modelling, prediction
from warnings import filterwarnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot
data_raw = fileRead.import_data()
data = preProcessing.processing_step1(data_raw)
X, y, X_train, X_test, y_train, y_test, X_test_copy = preProcessing.processing_step2(data)
model = Sequential()
model.add(Dense(units = 10, kernel_initializer = 'uniform',activation='relu', input_dim = X.shape[1]))
model.add(Dense(units = 10,activation='relu'))
model.add(Dense(units=1, activation = 'relu'))
model.compile(optimizer='adam', loss = 'mean_squared_error',metrics=['mse'])
history =model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=2, batch_size=30, verbose=1)
y_pred =model.predict(X_test)### plot loss during training
pyplot.title('mean_squared_error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
data = pd.DataFrame(y_pred)
data2 = pd.DataFrame(y_test)
data.to_excel('sample_data.xlsx', sheet_name='sheet1', index=False)
data2.to_excel('sample_data2.xlsx', sheet_name='sheet1', index=False)