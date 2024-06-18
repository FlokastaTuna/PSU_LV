import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def funkcija(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def dodaj_noise(y):
    np.random.seed(14)
    buka = np.max(y) - np.min(y)
    y_bucni = y + 0.1*buka*np.random.normal(0,1,len(y))
    return y_bucni
 
x = np.linspace(1,10,50)
y_istinit = funkcija(x)
y_izmjeren = dodaj_noise(y_istinit)

x = x[:, np.newaxis]
y_izmjeren = y_izmjeren[:, np.newaxis]


polinom = PolynomialFeatures(degree=15)
x_novi = polinom.fit_transform(x)
    
np.random.seed(12)
indeksi = np.random.permutation(len(x_novi))
indeksi_trening = indeksi[0:int(np.floor(0.7*len(x_novi)))]
indeksi_testni = indeksi[int(np.floor(0.7*len(x_novi)))+1:len(x_novi)]

x_trenirani = x_novi[indeksi_trening,]
y_trenirani = y_izmjeren[indeksi_trening]

x_testni = x_novi[indeksi_testni,]
y_testni = y_izmjeren[indeksi_testni]

linearModel = lm.LinearRegression()
linearModel.fit(x_trenirani,y_trenirani)

y_testnipolinom = linearModel.predict(x_testni)
mse_testirani = mean_squared_error(y_testni, y_testnipolinom)

plt.figure(1)
plt.plot(x_testni[:,1],y_testnipolinom,'og',label='predicted')
plt.plot(x_testni[:,1],y_testni,'or',label='test')
plt.legend(loc = 4)
plt.show()


plt.figure(2)
plt.plot(x,y_istinit,label='f')
plt.plot(x, linearModel.predict(x_novi),'r-',label='model')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_trenirani[:,1],y_trenirani,'ok',label='train')
plt.legend(loc = 4)
plt.show()
