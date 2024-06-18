import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cars_processed.csv')


print(df.info())


sns.pairplot(df, hue='fuel')

sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name','mileage'], axis=1)

obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()


fig = plt.figure(figsize=[15,8])
for i, col in enumerate(obj_cols):
    plt.subplot(2, 2, i + 1)
    sns.countplot(x=col, data=df)


plt.figure(figsize=[10,6])
sns.boxplot(x='fuel', y='selling_price', data=df)

plt.figure(figsize=[8,5])
df['selling_price'].hist(grid=False)

matrica_zabune = df[num_cols].corr()
sns.heatmap(matrica_zabune, annot=True, linewidths=2, cmap='coolwarm')

plt.show()

