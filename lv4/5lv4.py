import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error


# ucitavanje podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())

x = df[['km_driven', 'year', 'engine', 'max_power']]
y = df['selling_price']

# podjela na train i test
x_trening, x_testiran, y_treniran, y_testiran = train_test_split(x, y, test_size=0.2, random_state=300)

skaliranje = StandardScaler()
x_trening_skaliran = skaliranje.fit_transform(x_trening)
x_testiran_skaliran = skaliranje.transform(x_testiran)

linear_model = LinearRegression()
linear_model.fit(x_trening_skaliran, y_treniran)

y_predikcija_trening = linear_model.predict(x_trening_skaliran)
y_predikcija_testiran = linear_model.predict(x_testiran_skaliran)

print("R2 test", r2_score(y_predikcija_testiran, y_testiran))
print("RMSE test:", np.sqrt(mean_squared_error(y_predikcija_testiran, y_testiran)))
print("Max error test:", max_error(y_predikcija_testiran, y_testiran))
print("MAE test:", mean_absolute_error(y_predikcija_testiran, y_testiran))
y_predikcija_rupe = np.exp(y_predikcija_testiran)
y_testiran_rupe = np.exp(y_testiran)
print("TRUE RMSE",np.sqrt(mean_squared_error(y_predikcija_rupe, y_testiran_rupe)))
print("TRUE MAE",mean_absolute_error(y_predikcija_rupe, y_testiran_rupe))

figura = plt.figure(figsize=[13, 10])
iks = sns.regplot(x=y_predikcija_testiran, y=y_testiran, line_kws={'color': 'green'})
iks.set(xlabel = 'Predikcija', ylabel = 'Stvarna vrijednost', title='Rezultati na testnim podacima')
plt.show()

#smanjenjem broja ulaznih veličina, preciznost modela pada
