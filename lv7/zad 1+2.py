import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



(x_trening, y_trening), (x_testni, y_testni) = keras.datasets.mnist.load_data()


plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_trening[i], cmap="gray")
    plt.title(f"Label: {y_trening[i]}")
    plt.axis('off')
plt.show()


x_trening_s = x_trening.astype("float32") / 255
x_testni_s = x_testni.astype("float32") / 255


x_trening_s = x_trening_s.reshape(60000, 784)
x_testni_s = x_testni_s.reshape(10000, 784)


y_trening_s = keras.utils.to_categorical(y_trening, 10)
y_testni_s = keras.utils.to_categorical(y_testni, 10)



x_trening_s = x_trening.astype("float32") / 255
x_testni_s = x_testni.astype("float32") / 255


x_trening_s = x_trening_s.reshape(60000, 784)
x_testni_s = x_testni_s.reshape(10000, 784)


y_trening_s = keras.utils.to_categorical(y_trening, 10)
y_testni_s = keras.utils.to_categorical(y_testni, 10)



model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()



model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])




povijest = model.fit(x_trening_s, y_trening_s, epochs=10, batch_size=128, validation_split=0.1)



treniranje_gubici, treniranje_tocnost = model.evaluate(x_trening_s, y_trening_s, verbose=0)
testiranje_gubici, testriranje_tocnost = model.evaluate(x_testni_s, y_testni_s, verbose=0)

print(f'Točnost na train skupu: {treniranje_tocnost:.4f}')
print(f'Točnost na test skupu: {testriranje_tocnost:.4f}')


y_predikcija = model.predict(x_testni_s)
y_predikcija_klase = np.argmax(y_predikcija, axis=1)
y_istina = np.argmax(y_testni_s, axis=1)

cm = confusion_matrix(y_istina, y_predikcija_klase)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.show()


incorrect_indices = np.where(y_predikcija_klase != y_istina)[0]

plt.figure(figsize=(10, 4))
for i, incorrect in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_testni[incorrect], cmap="gray")
    plt.title(f"Pred: {y_predikcija_klase[incorrect]}, Prava: {y_istina[incorrect]}")
    plt.axis('off')
plt.show()

#matrica zabune pokazuje da se vecina brojeva dobro klasificirala
# klasifikator radio greške koje su bile vrlo vjerovatne za dogoditi se, mijesanje okruglih brojeva 9 0 i 6
