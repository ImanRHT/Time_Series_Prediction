################
# Code from OP #
################
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.layers import Dropout
import numpy as np
import json
data = open("EU_D1.json", "r")
data = json.load(data)

op = data["open"]
hi = data["high"]
lo = data["low"]
cl = data["close"]

open1 = []
high1 = []
low1 = []
close1 = []



for i in range(0, 2000):
    open1.append(float(op[i]))
    high1.append(float(hi[i]))
    low1.append(float(lo[i]))
    close1.append(float(cl[i]))

open1 = np.array(open1)
high1 = np.array(high1)
low1 = np.array(low1)
close1 = np.array(close1)


def random_sample(len_timeseries=2000):
    data = open("EU_D1.json", "r")
    data = json.load(data)
    op = data["open"]
    hi = data["high"]
    lo = data["low"]
    cl = data["close"]

    open1 = []
    high1 = []
    low1 = []
    close1 = []



    for i in range(0, len_timeseries):
        open1.append(float(op[i]))
        high1.append(float(hi[i]))
        low1.append(float(lo[i]))
        close1.append(float(cl[i]))

    open1 = np.array(open1)
    high1 = np.array(high1)
    low1 = np.array(low1)
    close1 = np.array(close1)

    x1 = open1#np.cos(np.arange(0,len_timeseries)/float(55.0))#+ np.random.choice(Nchoice)))
    x2 = high1#np.cos(np.arange(0,len_timeseries)/float(55.0))# + np.random.choice(Nchoice)))
    x3 = low1#np.sin(np.arange(0,len_timeseries)/float(55.0))# + np.random.choice(Nchoice)))
    x4 = close1#np.sin(np.arange(0,len_timeseries)/float(55.0))# + np.random.choice(Nchoice)))

    y1 = np.random.random(len_timeseries)
    y2 = np.random.random(len_timeseries)
    y3 = np.random.random(len_timeseries)
    y4 = np.random.random(len_timeseries)

    for t in range(3,len_timeseries):
        ## the output time series depend on input as follows:
        y1[t] = x1[t]
        y2[t] = x2[t]
        y3[t] = x3[t]
        y4[t] = x4[t]

    y = np.array([y1,y2,y3,y4]).T
    X = np.array([x1,x2,x3,x4]).T

    return y, X

def generate_data(Nsequence):
    X_train = []
    y_train = []
    y, X = random_sample()
    for isequence in range(Nsequence):

        X_train.append(X)
        y_train.append(y)
    return np.array(X_train),np.array(y_train)

Nsequence = 1000
prop = 0.5
Ntrain = int(Nsequence*prop)
X, y = generate_data(Nsequence)
print(X.shape)
print(y.shape)



X_train = X[:Ntrain,:,:]
X_test  = X[Ntrain:,:,:]
y_train = y[:Ntrain,:,:]
y_test  = y[Ntrain:,:,:]


#X.shape = (N sequence, length of time series, N input features)
#y.shape = (N sequence, length of time series, N targets)
print(X.shape, y.shape)
# (100, 3000, 4) (100, 3000, 3)
print(X_train.shape, y_train.shape)
####################
# Cutting function #
####################
def stateful_cut(arr, batch_size, T_after_cut):
    if len(arr.shape) != 3:
        # N: Independent sample size,
        # T: Time length,
        # m: Dimension
        print("ERROR: please format arr as a (N, T, m) array.")

    N = arr.shape[0]
    T = arr.shape[1]

    # We need T_after_cut * nb_cuts = T
    nb_cuts = int(T / T_after_cut)
    if nb_cuts * T_after_cut != T:
        print("ERROR: T_after_cut must divide T")

    # We need batch_size * nb_reset = N
    # If nb_reset = 1, we only reset after the whole epoch, so no need to reset
    nb_reset = int(N / batch_size)
    if nb_reset * batch_size != N:
        print("ERROR: batch_size must divide N")

    # Cutting (technical)
    cut1 = np.split(arr, nb_reset, axis=0)
    cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]
    cut3 = [np.concatenate(x) for x in cut2]
    cut4 = np.concatenate(cut3)
    return(cut4)

#############
# Main code #
#############
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

##
# Data
##
print( X_train.shape[0])
N = X_train.shape[0] # size of samples
T = X_train.shape[1] # length of each time series
batch_size = N # number of time series considered together: batch_size | N
T_after_cut = 100 # length of each cut part of the time series: T_after_cut | T
dim_in = X_train.shape[2] # dimension of input time series
dim_out = y_train.shape[2] # dimension of output time series

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


inputs, outputs, inputs_test, outputs_test =[stateful_cut(arr, batch_size, T_after_cut) for arr in [X_train, y_train, X_test, y_test]]

##
# Model
##
nb_units = 10

model = Sequential()
model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),
               return_sequences=True, units=nb_units, stateful=True))
model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model.compile(loss = 'mse', optimizer = 'adam')

##
# Training
##
epochs =300
nb_reset = int(N / batch_size)
if nb_reset > 1:
    print("ERROR: We need to reset states when batch_size < N")

# When nb_reset = 1, we do not need to reinitialize states
history = model.fit(inputs, outputs, epochs = epochs,
                    batch_size = batch_size, shuffle=False,
                    validation_data=(inputs_test, outputs_test))

def plotting(history):
    plt.plot(history.history['loss'], color = "red")
    plt.plot(history.history['val_loss'], color = "blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()

plt.figure(figsize=(10,8))
plotting(history) # Evolution of training/test loss

##
# Visual checking for a time series
##
## Mime model which is stateless but containing stateful weights
model_stateless = Sequential()
model_stateless.add(LSTM(input_shape=(None, dim_in),
               return_sequences=True, units=nb_units))
model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
model_stateless.compile(loss = 'mse', optimizer = 'adam')
model_stateless.set_weights(model.get_weights())

## Prediction of a new set
i = 0 # time series selected (between 0 and N-1)
x = X_train[i]
y = y_train[i]



print(X_train.shape)
print(y_train.shape)
print(x.shape)
print(y.shape)
y_hat = model_stateless.predict(np.array([x]))[0]
print(y_hat.shape)


#for dim in range(3): # dim = 0 for y1 ; dim = 1 for y2 ; dim = 2 for y3.
#    plt.figure(figsize=(10,8))
#    plt.plot(range(T), y[:,dim])
#    plt.plot(range(T), y_hat[:,dim])


def compire(test_data, pred_data):
    test_data = test_data
    pred_data = pred_data
    non_val_count = 0
    p_count_val = 0
    n_count_val = 0
    plt.plot(test_data,'--',label='Actual')
    plt.plot(pred_data,'-',label='Actual')
    #plt.show()
    for i in range(len(pred_data)):
        if (test_data[i-1]>test_data[i]) and (pred_data[i-1]>pred_data[i]):
            p_count_val = p_count_val + 1
        elif (test_data[i-1]<test_data[i]) and (pred_data[i-1]<pred_data[i]):
            n_count_val = n_count_val + 1
        else:
            non_val_count = non_val_count +1
        d = (str(((p_count_val+n_count_val)/200)*100))
    return p_count_val,n_count_val,non_val_count,d


plt.figure(figsize=(10,8))
plt.plot(open1,'-',label='LSTM')
plt.plot(high1,'-',label='LSTM')
plt.plot(low1,'-',label='LSTM')
plt.plot(close1,'-',label='LSTM')
#plt.plot(range(T), y[:,0],':')
plt.plot(range(T), y_hat[:,0],'--') #open
#plt.plot(range(T), y[:,1],'+')
plt.plot(range(T), y_hat[:,1],'--') #high
#plt.plot(range(T), y[:,2],':')
plt.plot(range(T), y_hat[:,2],'--') #low
#plt.plot(range(T), y[:,3],':')
plt.plot(range(T), y_hat[:,3],'--') #close


plt.show()
## Conclusion: works almost perfectly.

a,b,c,d = compire(open1,y_hat[2:,0])
print("acc:open")
print(float(d)/5)

a,b,c,d = compire(high1,y_hat[2:,1])
print("acc:high")
print(float(d)/5)

a,b,c,d = compire(high1,y_hat[2:,2])
print("acc:low")
print(float(d)/5)

a,b,c,d = compire(high1,y_hat[2:,3])
print("acc:close")
print(float(d)/5)
