import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error


df = pd.read_csv('cars_processed.csv')
print(df.info())

kategorije = ['fuel', 'seller_type', 'transmission', 'owner']


enkodiran = pd.get_dummies(df, columns=kategorije, drop_first=True)

x = enkodiran.drop('selling_price', axis=1)
y = enkodiran['selling_price']

numericke_vrijednosti = x.select_dtypes(include=['int64', 'float64']).columns
x_numericki = x[numericke_vrijednosti]


x_treniran, x_testiran, y_treniran, y_testiran = train_test_split(x_numericki, y, test_size=0.2, random_state=300)

skaliran = StandardScaler()
x_treniran_skaliran = skaliran.fit_transform(x_treniran)
x_test_skaliran = skaliran.transform(x_treniran)


linear_model = LinearRegression()
linear_model.fit(x_treniran_skaliran, y_treniran)

# Evaluacija modela
y_pred_train = linear_model.predict(x_treniran_skaliran)
y_pred_test = linear_model.predict(x_test_skaliran)

print("R2 test", r2_score(y_pred_test, y_testiran))
print("RMSE test:", np.sqrt(mean_squared_error(y_pred_test, y_testiran)))
print("Max error test:", max_error(y_pred_test, y_testiran))
print("MAE test:", mean_absolute_error(y_pred_test, y_testiran))

# Plot rezultata na testnim podacima
fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_testiran, line_kws={'color': 'green'})
ax.set(xlabel='Predikcija', ylabel='Stvarna vrijednost', title='Rezultati na testnim podacima')
plt.show()

#nije doslo do poboljsanja rezultata