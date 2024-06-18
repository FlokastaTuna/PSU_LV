from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os


(x_trenirani, y_trenirani), (x_testirani, y_testirani) = keras.datasets.mnist.load_data()
x_trenirani_s = x_trenirani.reshape(-1, 28, 28, 1) / 255.0
x_testirani_s = x_testirani.reshape(-1, 28, 28, 1) / 255.0

y_trenirani_s = to_categorical(y_trenirani, num_classes=10)
y_testirani_s = to_categorical(y_testirani, num_classes=10)


model = models.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


log_dir = "logs"
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

povratni_poziv = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(x_trenirani_s, y_trenirani_s,
                    batch_size=128,
                    epochs=10,
                    validation_split=0.1,
                    callbacks=[model_checkpoint_callback, povratni_poziv])


najbolji_model = models.load_model('best_model.keras')


trening_gubici, trening_tocnost = najbolji_model.evaluate(x_trenirani_s, y_trenirani_s, verbose=0)
testiran_gubici, testiran_tocnost = najbolji_model.evaluate(x_testirani_s, y_testirani_s, verbose=0)

print(f'Točnost na train skupu: {trening_tocnost:.4f}')
print(f'Točnost na test skupu: {testiran_tocnost:.4f}')


y_predikcija = najbolji_model.predict(x_testirani_s)
y_predikcija_klasa = np.argmax(y_predikcija, axis=1)
y_istina = np.argmax(y_testirani_s, axis=1)

matrica = confusion_matrix(y_istina, y_predikcija_klasa)
disp = ConfusionMatrixDisplay(confusion_matrix=matrica, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.show()


y_trening_predikcija = najbolji_model.predict(x_trenirani_s)
y_trening_predikcija_klase = np.argmax(y_trening_predikcija, axis=1)
y_trening_istina = np.argmax(y_trenirani_s, axis=1)

matrica_treninga = confusion_matrix(y_trening_istina, y_trening_predikcija_klase)
pokazi_trening = ConfusionMatrixDisplay(confusion_matrix=matrica_treninga, display_labels=range(10))
pokazi_trening.plot(cmap=plt.matrica.Blues)
plt.show()

#matrica zabune prikazuju visoku točnost rezultata