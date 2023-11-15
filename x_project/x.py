import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import json
import keras



save_model = False
load_model = False
EPOCH = 50
window = W = 200
type = "close"


MODEL_NAME = "xyz"
MODEL_NAME1 = "X_open_W200_5000"
MODEL_NAME2 = "X4_high_W200_5000"
MODEL_NAME3 = "X5_low_W200_5000"
MODEL_NAME4 = "X2_close_W200_5000"
#MODEL_NAME = "X2_"+type+"_W"+str(W)+"_"+str(EPOCH)


def compire(test_data, pred_data):
    test_data = test_data
    pred_data = pred_data
    non_val_count = 0
    p_count_val = 0
    n_count_val = 0
    plt.plot(test_data,'--',label='Actual')
    plt.plot(pred_data,'-',label='Actual')
    plt.show()
    for i in range(len(pred_data)):
        if (test_data[i-1]>test_data[i]) and (pred_data[i-1]>pred_data[i]):
            p_count_val = p_count_val + 1
        elif (test_data[i-1]<test_data[i]) and (pred_data[i-1]<pred_data[i]):
            n_count_val = n_count_val + 1
        else:
            non_val_count = non_val_count +1
        d = (str(((p_count_val+n_count_val)/200)*100)+"%")
    return p_count_val,n_count_val,non_val_count,d





data = open("EU_D1.json", "r")
data = json.load(data)

open = data["open"]
high = data["high"]
low = data["low"]
close = data["close"]

open1 = []
open2 = []
high1 = []
high2 = []
low1 = []
low2 = []
close1 = []
close2 = []
val_close = []

for i in range(500, 1500):
    open1.append(float(open[i]))
    high1.append(float(high[i]))
    low1.append(float(low[i]))
    close1.append(float(close[i]))

for i in range(1300, 1500):
    open2.append(float(open[i]))
    high2.append(float(high[i]))
    low2.append(float(low[i]))
    close2.append(float(close[i]))

for i in range(1000,1200):
    val_close.append(float(close[i]))

XX = close2
X = close1
www = XX
last = int(len(X)/5.0)
open1 = np.array(open1)
open2 = np.array(open2)
high1 = np.array(high1)
high2 = np.array(high2)
low1 = np.array(low1)
low2 = np.array(low1)
close1 = np.array(close1)
close2 = np.array(close2)
Xtest_open = open1[-last-window:]
Xtest_high = high1[-last-window:]
Xtest_low = low1[-last-window:]
Xtest_close = close1[-last-window:]


################# DATA_PREPARATION #################
def generate_seq(window,data):
    last = int(len(data)/5.0)
    Xtrain = data[:-last]
    Xtest = data[-last-window:]
    xin = []
    next_X = []
    for i in range(window,len(Xtrain)):
        xin.append(Xtrain[i-window:i])
        next_X.append(Xtrain[i])
    xin, next_X = np.array(xin), np.array(next_X)
    xin = xin.reshape(xin.shape[0], xin.shape[1], 1)
    return xin,next_X,Xtrain,Xtest
#xin,next_X,Xtrain,Xtest = generate_seq(window,X)

def matrix_generate_seq(window,data,mod=0):
    days = []
    n_day = 1000
    for i in range(n_day):
        day = [data["open"][i],data["high"][i],data["low"][i],data["close"][i]]
        days.append(day)
    days = np.array(days).reshape(n_day,4,1)
    day = np.array(day).reshape(4,1)
    last = int(n_day/5.0)
    Xtrain = days[:-last]
    Xtest = days[-last-window:]
    xin = []
    next_X = []
    if mod == 0:
        x = Xtrain
    elif mod == 1:
        x = Xtest
    for i in range(window,len(x)):
        xin.append(x[i-window:i])
        next_X.append(x[i])
    xin, next_X = np.array(xin), np.array(next_X)
    xin = xin.reshape(xin.shape[0], xin.shape[1],4)
    next_X = next_X.reshape(next_X.shape[0],4)
    print("===================")
    print(day.shape)
    print(days.shape)
    print(Xtest.shape)
    print(Xtrain.shape)
    print(xin.shape)
    print(next_X.shape)
    print("===================")
    return xin,next_X,Xtrain,Xtest


xin,next_X,Xtrain,Xtest = matrix_generate_seq(window,data,0)
print("++++++++++++")
print(xin.shape)
print(next_X.shape)
print(Xtrain.shape)
print(Xtest.shape)
print("++++++++++++")


Xval = X[-last:] #800-1000

########## Initialize LSTM model ###############
if load_model == True:
    m1 = keras.models.load_model(MODEL_NAME1+".h5")
    m2 = keras.models.load_model(MODEL_NAME2+".h5")
    m3 = keras.models.load_model(MODEL_NAME3+".h5")
    m4 = keras.models.load_model(MODEL_NAME4+".h5")

else:
    m = Sequential()
    m.add(LSTM(units=50, return_sequences=True, input_shape=(200,4)))
    m.add(Dropout(0.2))
    m.add(LSTM(units=50))
    m.add(Dropout(0.2))
    m.add(Dense(units=1))
    m.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fit LSTM model
    history = m.fit(xin, next_X, epochs = EPOCH, batch_size = 100,verbose=0)
    if save_model == True:
        m.save(MODEL_NAME+".h5")

xin,next_X,Xtrain,Xtest = matrix_generate_seq(window,data,1)

print("++++++++++++")
print(xin.shape)
print(next_X.shape)
print(Xtrain.shape)
print(Xtest.shape)
print("++++++++++++")


X_pred = m.predict(xin)
print(xin.shape)
print(X_pred.shape)
plt.plot(next_X,'--',label='Actual')
plt.plot(X_pred,'--',label='Actual')
plt.show()

xin_open = []
xin_high = []
xin_low = []
xin_close = []
next_X_open = []
next_X_high = []
next_X_low = []
next_X_close = []

for i in range(window,len(Xtest)):
    xin_open.append(Xtest_open[i-window:i])
    next_X_open.append(Xtest_open[i])
xin_open, next_X_open = np.array(xin_open), np.array(next_X_open)
xin_open = xin_open.reshape((xin_open.shape[0], xin_open.shape[1], 1))

for i in range(window,len(Xtest)):
    xin_high.append(Xtest_high[i-window:i])
    next_X_high.append(Xtest_high[i])
xin_high, next_X_high = np.array(xin_high), np.array(next_X_high)
xin_high = xin_high.reshape((xin_high.shape[0], xin_high.shape[1], 1))

for i in range(window,len(Xtest)):
    xin_low.append(Xtest_low[i-window:i])
    next_X_low.append(Xtest_low[i])
xin_low, next_X_low = np.array(xin_low), np.array(next_X_low)
xin_low = xin_low.reshape((xin_low.shape[0], xin_low.shape[1], 1))

for i in range(window,len(Xtest)):
    xin_close.append(Xtest_close[i-window:i])
    next_X_close.append(Xtest_close[i])
xin_close, next_X_close = np.array(xin_close), np.array(next_X_close)
xin_close = xin_close.reshape((xin_close.shape[0], xin_close.shape[1], 1))


X_pred_open   = m1.predict(xin_open) #201-400
X_pred_high   = m2.predict(xin_high)
X_pred_low    = m3.predict(xin_low)
X_pred_close  = m4.predict(xin_close)

# 2 day predict
xinn1=xin_open
xinn2=xin_high
xinn3=xin_low
xinn4=xin_close
last_pred1 = X_pred_open[-1]
xin_open = xin_open[1:]
xin_open_last_window = xin_open[-1]
last_window_xin = np.concatenate((xin_open_last_window[1:] , last_pred1.reshape(1,1)))
xin5 = np.concatenate((xin_open , last_window_xin.reshape(1,200,1)))
X_pred2_open = m1.predict(xin5)


last_pred2 = X_pred_high[-1]
xin_high = xin_high[1:]
xin_high_last_window = xin_high[-1]
last_window_xin = np.concatenate((xin_high_last_window[1:] , last_pred2.reshape(1,1)))
xin6 = np.concatenate((xin_high , last_window_xin.reshape(1,200,1)))
X_pred2_high = m2.predict(xin6)

last_pred3 = X_pred_low[-1]
xin_low = xin_low[1:]
xin_low_last_window = xin_low[-1]
last_window_xin = np.concatenate((xin_low_last_window[1:] , last_pred3.reshape(1,1)))
xin7 = np.concatenate((xin_low , last_window_xin.reshape(1,200,1)))
X_pred2_low = m3.predict(xin7)









'''

val_2d = []
for i in range(100):
    x = m4.predict(xin_close)[-1][-1][-1]
    val = x
    val_2d.append(val)
    xin_close = xin_close[1:]
    xin_close_last_window = xin_close[-1]
    last_window_xin = np.concatenate((xin_close_last_window[1:] , val.reshape(1,1)))
    new_xin_close = np.concatenate((xin_close , last_window_xin.reshape(1,200,1)))



'''

last_pred4 = X_pred_close[-1]
xin_close = xin_close[1:]
xin_close_last_window = xin_close[-1]
last_window_xin = np.concatenate((xin_close_last_window[1:] , last_pred4.reshape(1,1)))
xin8 = np.concatenate((xin_close , last_window_xin.reshape(1,200,1)))
X_pred2_close = m4.predict(xin8)




min = []
for i in range(len(X_pred_open)):
    min = ((X_pred_close)+(X_pred2_close))/2
min = np.array(min)


# Plot prediction vs actual for test data
plt.figure()
#plt.plot(X_pred_open,'--',label='LSTM')
#plt.plot(X_pred_high,'--',label='LSTM')
#plt.plot(X_pred_low,'--',label='LSTM')

#plt.plot(X_pred3_close,'--',label='LSTM')
#plt.plot(X_pred2_open,'-',label='e')
#plt.plot(X_pred2_high,'-',label='e')
#plt.plot(X_pred2_low,'-',label='w')
#plt.plot(X_pred2_close,'-',label='s')
#plt.plot(next_X_open,'-',label='Actual')
#plt.plot(next_X_high,'-',label='Actual')
#plt.plot(next_X_low,'-',label='Actual')
#plt.plot(next_X_close,'-',label='Actual')
plt.plot(X_pred_close,':',label='LSTM')
plt.plot(X_pred2_close,'--',label='LSTM')
plt.plot(www,'-',label='Actual')
#plt.plot(next_X_high,'+',label='Actual')
#plt.plot(next_X_low,'+',label='Actual')
#plt.plot(next_X_close,'+',label='Actual')
plt.plot(min,'--',label='Actual')
plt.show()
plt.savefig(MODEL_NAME+"_0.png")







a,b,c,d = compire(next_X_close,X_pred_open)
print("open")
print(a)
print(b)
print(c)
print(d)


a,b,c,d = compire(next_X_high,X_pred_high)
print("high")
print(a)
print(b)
print(c)
print(d)


a,b,c,d = compire(next_X_low,X_pred_low)
print("low")
print(a)
print(b)
print(c)
print(d)


a,b,c,d = compire(next_X_close,X_pred_close)
print("close")
print(a)
print(b)
print(c)
print(d)

a,b,c,d = compire(next_X_close,X_pred2_close)
print("close")
print(a)
print(b)
print(c)
print(d)

a,b,c,d = compire(next_X_close,min)
print("mean")
print(a)
print(b)
print(c)
print(d)

'''
a,b,c,d = compire(next_X_open,min)
print("min")
print(a)
print(b)
print(c)
print(d)
a,b,c,d = compire(next_X_high,min)
print("min")
print(a)
print(b)
print(c)
print(d)
a,b,c,d = compire(next_X_low,min)
print("min")
print(a)
print(b)
print(c)
print(d)
a,b,c,d = compire(next_X_close,min)
print("min")
print(a)
print(b)
print(c)
print(d)

'''





'''

aaa = open("Validation_xyz_MIN.txt","a")

aaa.write("p_count_val = "+str(p_count_val)+"\n"+
            "n_count_val = "+str(n_count_val)+"\n"+
            "non_val_count = "+str(non_val_count)+"\n"+
            "acc = "+str(((p_count_val+n_count_val)/200)*100)+"%"+"\n"+
            str(non_val_pointer))
aaa.close()








X_pred = X_pred.reshape(200,)
qval = np.array(qval)
qval = np.concatenate((X_pred , qval.reshape(200,)))

plt.figure()
plt.plot(qval,'-',label='LSTM')
plt.plot(www,'--',label='Actual')
plt.savefig(MODEL_NAME+"_1.png")






for i in range(200):
    pred = m.predict(xin) #xin:(200,200,1)
    pred = pred[-1]

    Xval = Xval[1:]
    Xval = np.concatenate((Xval , pred))
    Xval = Xval.reshape(1,200,1)

    xin = xin[1:] #xin:(199,200,1)
    xin = np.concatenate((xin , Xval))

    Xval = Xval.reshape(200,)

X_pred = X_pred.reshape(200,)
Xval = np.concatenate((X_pred , Xval))

plt.figure()
plt.plot(Xval,'-',label='LSTM')
plt.plot(www,'--',label='Actual')
plt.savefig(MODEL_NAME+"_2.png")


'''
