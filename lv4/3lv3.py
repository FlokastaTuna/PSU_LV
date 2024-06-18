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

stupnjevi = [2, 6, 15]

mse_treniran = []
mse_testiran = []

plt.figure(figsize=(15, 5))

for i, stupnjevi in enumerate(stupnjevi):
    
    polinom = PolynomialFeatures(stupnjevi=stupnjevi)
    x_novi = polinom.fit_transform(x)
        
    np.random.seed(12)
    indeksi = np.random.permutation(len(x_novi))
    indeksi_trening = indeksi[0:int(np.floor(0.7*len(x_novi)))]
    indeksi_test = indeksi[int(np.floor(0.7*len(x_novi)))+1:len(x_novi)]

    x_treniran = x_novi[indeksi_trening,]
    y_treniran = y_izmjeren[indeksi_trening]

    x_testiran = x_novi[indeksi_test,]
    y_testiran = y_izmjeren[indeksi_test]

    linearModel = lm.LinearRegression()
    linearModel.fit(x_treniran,y_treniran)

    y_trenirani_polinom = linearModel.predict(x_treniran)
    y_testirani_polinom = linearModel.predict(x_testiran)

    mse_treniran.append(mean_squared_error(y_treniran, y_trenirani_polinom))
    mse_testiran.append(mean_squared_error(y_testiran, y_testirani_polinom))

    plt.subplot(1, len(stupnjevi), i+1)
    plt.plot(x_testiran[:,1], y_testirani_polinom, 'og', label='predicted')
    plt.plot(x_testiran[:,1], y_testiran, 'or', label='test')
    plt.plot(x, y_istinit, label='true function')
    plt.title(f'Degree = {stupnjevi}')
    plt.legend()

plt.show()

print("MSE train:", mse_treniran)
print("MSE test:", mse_testiran)

#ako imamo veći broj uzoraka za učenje, zavrsni model će biti blizi idealnom rješenju
#ako imamo veći broj uzoraka za učenje, model je manje precizan