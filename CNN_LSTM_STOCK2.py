from collections.abc import Sequence
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.
    target_type = df[target].dtypes
    target_type = target_type[0] if isinstance(target_type, Sequence) else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Regression chart.
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=0, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low


def to_sequences(seq_size, data1 ,data2):
    print(data1.shape)
    print(data1)
    x = []
    y = []

    for i in range(len(data1)-SEQUENCE_SIZE-1):
        #print(i)
        window = data1[i:(i+SEQUENCE_SIZE)]
        after_window = data2[i+SEQUENCE_SIZE]
        #window = [[x] for x in window]
       #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)

    return np.array(x),np.array(y)

def to_sequences_cnn(seq_size, data1 ,data2):
    x = []
    y = []

    for i in range(len(data1)-SEQUENCE_SIZE-1):
        #print(i)
        window = data1[i:(i+SEQUENCE_SIZE)]
        after_window = data2[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
       #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)

    return np.array(x),np.array(y)

import pandas as pd



df = pd.read_csv('data/CSC215_P2_Stock_Price.csv')
df.drop(['Date','Adj_Close'], axis=1, inplace=True)
df = df.dropna() #drop any null value row





plt.figure()
plt.plot(df["Close"],'-',label='LSTM')
plt.plot(df["Open"],'--',label='Actual')
plt.plot(df["High"],'--',label='Actual')
plt.plot(df["Low"],'--',label='Actual')
plt.show()

df_stock_close = df['Close'].tolist()



encode_numeric_range(df, 'Close')
encode_numeric_range(df, 'Open')
encode_numeric_range(df, 'High')
encode_numeric_range(df, 'Low')
encode_numeric_range(df, 'Volume')


plt.figure()
plt.plot(df["Close"],'-',label='LSTM')
plt.plot(df["Open"],'--',label='Actual')
plt.plot(df["High"],'--',label='Actual')
plt.plot(df["Low"],'--',label='Actual')
plt.show()


#Preparing x and y
SEQUENCE_SIZE = 7
x,y = to_sequences(SEQUENCE_SIZE, df.values, df_stock_close)
print("Shape of x: {}".format(x.shape))
print("Shape of y: {}".format(y.shape))
x_NN = x.reshape(len(df)-SEQUENCE_SIZE-1,SEQUENCE_SIZE*5)
y_NN = y



x_train_NN,x_test_NN,y_train_NN,y_test_NN = train_test_split(x_NN,y_NN, test_size=0.3,random_state=42)
print("Shape of x_train: {}".format(x_train_NN.shape))
print("Shape of x_test: {}".format(x_test_NN.shape))
print("Shape of y_train: {}".format(y_train_NN.shape))
print("Shape of y_test: {}".format(y_test_NN.shape))


# Load modules
import io
import requests
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras import regularizers


save_path = "./dnn/"

checkpointer = ModelCheckpoint(filepath="dnn/best_weights_NN.hdf5", verbose=0, save_best_only=True) # save best model

for i in range(5):
    model = Sequential()
    model.add(Dense(150,input_dim=x_train_NN.shape[1], activation='relu'))   # hidden 1
    model.add(Dropout(0.10))
    model.add(Dense(100,activation='relu')) # Hidden 2
    model.add(Dropout(0.10))
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(1)) # Output
    model.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=1, mode='auto')
    model.fit(x_train_NN,y_train_NN,validation_data=(x_test_NN,y_test_NN),callbacks=[monitor,checkpointer],verbose=2,epochs=1000)


from sklearn import metrics
from sklearn.metrics import r2_score

model.load_weights('dnn/best_weights_NN.hdf5')
neural_pred = model.predict(x_test_NN)
score = np.sqrt(metrics.mean_squared_error(y_test_NN,neural_pred))

print("RMSE         : {}".format(score))
print("MSE          :", metrics.mean_squared_error(y_test_NN, neural_pred))
print("R2 score     :",metrics.r2_score(y_test_NN,neural_pred))


chart_regression(neural_pred.flatten(),y_test_NN)
chart_regression(neural_pred.flatten(),y_test_NN,sort=False)
