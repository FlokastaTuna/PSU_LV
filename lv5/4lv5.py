import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('occupancy_processed.csv')

imena_potrebna = ['S3_Temp', 'S5_CO2']
imena_meta = 'Room_Occupancy_Count'
imena_klasa = ['Slobodna', 'Zauzeta']

x = df[imena_potrebna].to_numpy()
y = df[imena_meta].to_numpy()


x_treniran, x_testiran, y_treniran, y_testiran = train_test_split(x, y, test_size=0.2, random_state=42)


regresija = LogisticRegression()
regresija.fit(x_treniran, y_treniran)


y_predikcija = regresija.predict(x_testiran)
print("Tocnost:", accuracy_score(y_testiran, y_predikcija))
print("izvjesce klasifikacije:")
print(classification_report(y_testiran, y_predikcija, target_names=imena_klasa))


plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(x[mask, 0], x[mask, 1], label=imena_klasa[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('zauzetost')
plt.legend()
plt.show()


#model koristi numeričke postotke ispravno klasificiranih od ukupnih ulaza i daje preciznost od 88%

