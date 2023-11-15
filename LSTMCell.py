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
		self.model.fit(X, y, epochs=1000, verbose=0)

		return self.model


	def model_predict(self, x_input):
		yhat = self.model.predict(x_input, verbose=0)

		return yhat



	def update_edge_history(self, momentary_load_level, t):
		self.history[0][t] = momentary_load_level

		return

	def get_history(self):
		return self.history
