
import matplotlib.pyplot as plt
import random

a = open("qqq.txt", 'w')
X = []

for i in range(40):
    a.write(str(random.uniform(0.75,0.76))+"\n")

#X = [0.045,0.225,0.504,0.71,0.82,0.912,0.932,0.949,0.962,0.973]

#[0.0,0.035,0.05,0.095,0.187,0.351,0.590,0.660,0.878,0.989]

Y = [0.1,0.135,0.05,0.095,0.187,0.351,0.590,0.660,0.878,0.989]




# Plot prediction vs actual for test data

plt.figure()
plt.plot(X,':',label='LSTM')
plt.plot(Y,'--',label='Actual')




# Plot prediction vs actual for test data

#plt.plot(X_pred,':',label='LSTM')
#plt.plot(next_X1,'--',label='Actual')

plt.show()

print("end")
