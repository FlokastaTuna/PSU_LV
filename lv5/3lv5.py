import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


df = pd.read_csv('occupancy_processed.csv')

imena_potrebna = ['S3_Temp', 'S5_CO2']
imena_meta = 'Room_Occupancy_Count'
imena_klasa = ['Slobodna', 'Zauzeta']

x = df[imena_potrebna].values
y = df[imena_meta].values

x_treniran, x_testiran, y_treniran, y_testiran = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


dt = DecisionTreeClassifier()
dt.fit(x_treniran, y_treniran)


y_predikcija = dt.predict(x_testiran)


print("Matrica zabune:")
print(confusion_matrix(y_testiran, y_predikcija))
print("Tocnost:", accuracy_score(y_testiran, y_predikcija))
print("Preciznost:", precision_score(y_testiran, y_predikcija, average=None))
print("Vracanje:", recall_score(y_testiran, y_predikcija, average=None))

plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=imena_potrebna, class_names=imena_klasa)
plt.show()

#promjenom parametar max-depth moze doci do jednostavnijeg ili kompliciranijeg stabla
#bez skaliranja stablo ima problem pronalazaka granica