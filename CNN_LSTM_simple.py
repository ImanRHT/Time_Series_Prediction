# univariate cnn lstm example
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import time
start_time = time.time()

class LSTMCell:
	def __init__(self):
		n_edgenode = 4
		n_history = 10
		self.model = self.create_model()
		self.H = np.zeros((n_history ,n_edgenode))
		self.history = np.zeros((1,n_history))

	def create_model(self):
		n_features = 1
		n_seq = 2
		n_steps = 2
		model = Sequential()
		model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
		model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
		model.add(TimeDistributed(Flatten()))
		model.add(LSTM(50, activation='relu'))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse')


		return model


	def model_fit(self, X, y):
		# fit model
		self.model.fit(X, y, epochs=500, verbose=0)

		return self.model


	def model_predict(self, x_input):
		yhat = self.model.predict(x_input, verbose=0)

		return yhat



	def update_edge_history(self, momentary_load_level, t):
		self.history[0][t] = momentary_load_level

		return

	def get_history(self):
		return self.history


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



A = LSTMCell()
B = LSTMCell()
C = LSTMCell()
D = LSTMCell()




for i in range(0, A.history.size):
	A.update_edge_history( i+1, i)

for i in range(0, B.history.size):
	B.update_edge_history( i+5, i)

for i in range(0, C.history.size):
	C.update_edge_history( i*3, i)

for i in range(0, D.history.size):
	D.update_edge_history( i*10, i)


h  = A.get_history()
h2 = B.get_history()
h3 = C.get_history()
h4 = D.get_history()

print(h)
print(h2)
print(h3)
print(h4)

n_steps = 10
# split into samples
X, y = split_sequence(h[0], n_steps)
X2,y2= split_sequence(h2[0], n_steps)
X3,y3= split_sequence(h3[0], n_steps)
X4,y4= split_sequence(h4[0], n_steps)

n_features = 1
n_seq = 5
n_steps = 2
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
print(X)

print(y)


print(X.shape)

print(y.shape)


X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
X2 = X2.reshape((X2.shape[0], n_seq, n_steps, n_features))
X3 = X3.reshape((X3.shape[0], n_seq, n_steps, n_features))
X4 = X4.reshape((X4.shape[0], n_seq, n_steps, n_features))


print(X.shape)

print(y.shape)



A.model_fit(X,y)
B.model_fit(X2,y2)
C.model_fit(X3,y3)
D.model_fit(X4,y4)

a = []
for i in range(0, 100):
	a.append(i)
b = []
for i in range(0, 100):
	b.append(i*5)
c = []
for i in range(0, 100):
	c.append(i*i)
d = []
for i in range(0, 100):
	d.append(i*10)


print(a)

x_input = array(a)
x_input = x_input.reshape((1, n_seq, n_steps, n_features))

x_input2 = array(b)
x_input2 = x_input2.reshape((1, n_seq, n_steps, n_features))

x_input3 = array(c)
x_input3 = x_input3.reshape((1, n_seq, n_steps, n_features))

x_input4 = array(d)
x_input4 = x_input4.reshape((1, n_seq, n_steps, n_features))

yhat = A.model_predict(x_input)
yhat2 = B.model_predict(x_input2)
yhat3 = C.model_predict(x_input3)
yhat4 = D.model_predict(x_input4)

print(yhat)
print(yhat2)
print(yhat3)
print(yhat4)
print("--- %s seconds ---" % (time.time() - start_time))
