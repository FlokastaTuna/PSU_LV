import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


df = pd.read_csv('occupancy_processed.csv')

imena_potrebna = ['S3_Temp', 'S5_CO2']
imena_meta = 'Room_Occupancy_Count'
imena_klasa = ['Slobodna', 'Zauzeta']

x = df[imena_potrebna].values
y = df[imena_meta].values


x_treniran, x_testiran, y_treniran, y_testiran = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

skaliranje = StandardScaler()
x_treniran_skaliran = skaliranje.fit_transform(x_treniran)
x_testiran_skaliran = skaliranje.transform(x_testiran)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_treniran_skaliran, y_treniran)

y_predikcija = knn.predict(x_testiran_skaliran)


print("Matrica zabune:")
print(confusion_matrix(y_testiran, y_predikcija))
print("Tocnost:", accuracy_score(y_testiran, y_predikcija))
print("Preciznost:", precision_score(y_testiran, y_predikcija, average=None))
print("Vracanje:", recall_score(y_testiran, y_predikcija, average=None))

plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=imena_klasa[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('zauzetost')
plt.legend()
plt.show()

#vise susjeda ili dovodi do preciznijeg rezultata ili dovodi do zanemarivanja uzoraka u podacima
#sa manjim brojem susjeda mozemo dobit bolje granice za ucenje, ali mozemo imati i underfitting i time losiju generalizacije.
#bez skaliranja model moze poceti prioritizirat vece velicine, a ignorirati manje i onda imamo one sided rezultate