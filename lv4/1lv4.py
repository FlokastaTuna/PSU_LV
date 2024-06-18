import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error

def funkcija(x):
	y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
	return y

def dodaj_noise(y):
    np.random.seed(14)
    buka = np.max(y) - np.min(y)
    y_bucni = y + 0.1*buka*np.random.normal(0,1,len(y))
    return y_bucni

x = np.linspace(1,10,100)
y_istinit = funkcija(x)
y_izmejeren = dodaj_noise(y_istinit)

plt.figure(1)
plt.plot(x,y_izmejeren,'ok',label='mjereno')
plt.plot(x,y_istinit,label='stvarno')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 4)
plt.show()

np.random.seed(12)
indeksi = np.random.permutation(len(x))
indeksi_trening = indeksi[0:int(np.floor(0.7*len(x)))]
indeksi_testni = indeksi[int(np.floor(0.7*len(x)))+1:len(x)]

x = x[:, np.newaxis]
y_izmejeren = y_izmejeren[:, np.newaxis]

x_treniran = x[indeksi_trening]
y_treniran = y_izmejeren[indeksi_trening]

x_testiran = x[indeksi_testni]
y_testiran = y_izmejeren[indeksi_testni]

plt.figure(2)
plt.plot(x_treniran,y_treniran,'ob',label='train')
plt.plot(x_testiran,y_testiran,'or',label='test')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 4)
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(x_treniran,y_testiran)

print('Model je oblika y_hat = Theta0 + Theta1 * x')
print('y_hat = ', linearModel.intercept_, '+', linearModel.coef_, '*x')

y_testiran_predikcija = linearModel.predict(x_treniran)
mse_test = mean_squared_error(y_testiran, y_testiran_predikcija)

plt.figure(3)
plt.plot(x_treniran,y_testiran_predikcija,'og',label='predicted')
plt.plot(x_treniran,y_testiran,'or',label='test')
plt.legend(loc = 4)

x_pravac = np.array([1,10])
x_pravac = x_pravac[:, np.newaxis]
y_pravac = linearModel.predict(x_pravac)
plt.plot(x_pravac, y_pravac)
plt.show()
