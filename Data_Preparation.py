import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


'''
data = open("EURUSD-D1.txt", "r")

price = data.read()

print(price)
f = price.split(",\n    ")
print(f)

k = []
for i in range(2000, 3000): ##len(f)-1):
    k.append(float(f[i]))


X = np.array(k)
'''

################# DATA_PREPARATION #################
n = 1000

t = np.linspace(0,20.0*np.pi,n)
X = np.sin(1000*t) # X is already between -1 and 1, scaling normally needed
# Set window of past points for LSTM model
'''
t = []
for i in range(0, n):
    t.append(i)
'''

X = np.array(X)
print(X)



#X = [1,2,3,4,5,6,7,8,9,10]
#X = np.array(t)

print(X)


window = 200
# Split 80/20 into train/test data
last = int(n/5.0)
Xtrain = X[:-last]
Xtest = X[-last-window:]

# Store window number of points as a sequence
xin = []
next_X = []
for i in range(window,len(Xtrain)):
    xin.append(Xtrain[i-window:i])
    next_X.append(Xtrain[i])

# Reshape data to format for LSTM
xin, next_X = np.array(xin), np.array(next_X)
xin = xin.reshape(xin.shape[0], xin.shape[1], 1)


#print(X)
print(X.shape)
print(Xtrain.shape)
print(Xtest.shape)
#print(xin)
print(xin.shape)
#print(next_X)
print(next_X.shape)




########## Initialize LSTM model ###############
m = Sequential()
m.add(LSTM(units=50, return_sequences=True, input_shape=(xin.shape[1],1)))
m.add(Dropout(0.2))
m.add(LSTM(units=50))
m.add(Dropout(0.2))
m.add(Dense(units=1))
m.compile(optimizer = 'adam', loss = 'mean_squared_error')


w = open("loss.text", 'a')

# Fit LSTM model
history = m.fit(xin, next_X, epochs = 50, batch_size = 100,verbose=0)

plt.figure()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.semilogy(history.history['loss'])

print(history.history['loss'])
w.write(str(history.history['loss']))


# Store "window" points as a sequence
xin = []
next_X1 = []
for i in range(window,len(Xtest)):
    xin.append(Xtest[i-window:i])
    next_X1.append(Xtest[i])





# Reshape data to format for LSTM
xin, next_X1 = np.array(xin), np.array(next_X1)

print(xin.shape)
print(next_X1.shape)


xin = xin.reshape((xin.shape[0], xin.shape[1], 1))

print(xin.shape)
# Predict the next value (1 step ahead)
X_pred = m.predict(xin)

print(X_pred.shape)
#print(Xtest.shape)

print(Xtest)
print("==============")
print(X_pred)


# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred,':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')
plt.semilogy(X_pred)
plt.semilogy(next_X1)


#abas = open("X_PRED.txt", 'a')

#for i in range(200):
#    abas.write(str(X_pred[i]) + '\n')

#abas2 = open("NEXT_X1.txt", 'a')

#for i in range(200):
#    abas2.write(str(next_X1[i]) + '\n')





# Using predicted values to predict next step
X_pred = Xtest.copy()
for i in range(window,len(X_pred)):
    xin = X_pred[i-window:i].reshape((1, window, 1))
    X_pred[i] = m.predict(xin)


print(Xtest)
print(X_pred)





# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred,':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')

plt.show()


print("end")
