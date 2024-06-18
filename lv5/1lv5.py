'''
Room occupancy classification 

R.Grbic, 2024.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('occupancy_processed.csv')

imena_potrebna = ['S3_Temp', 'S5_CO2']
imena_meta = 'Room_Occupancy_Count'
imena_klasa = ['Slobodna', 'Zauzeta']

X = df[imena_potrebna].to_numpy()
y = df[imena_meta].to_numpy()

# Scatter plot
plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=imena_klasa[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('zauzetost')
plt.legend()
plt.show()

#temperatura u zazetim sobama je visa u prosjeku, a povecanjem CO2 raste temperatura unutar istih
#skup podataka ima 10129 primjera, skup je podjeljen na zauzete i slobodne prostorije