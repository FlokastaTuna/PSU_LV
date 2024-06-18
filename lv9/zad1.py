import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

trening_direktorij = 'E:\Opcenito stvari\Faks-stručni\2.god\4.semestar\PSU\Labosi\lv9\Train'
testiran_direktorij = 'E:\Opcenito stvari\Faks-stručni\2.god\4.semestar\PSU\Labosi\lv9\Test'

trening_podaci = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

testirani_podaci = ImageDataGenerator(rescale=1./255)

treniran_generator = trening_podaci.flow_from_directory(
    trening_direktorij,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

provjeravani_podaci = trening_podaci.flow_from_directory(
    trening_direktorij,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

testirani_generator = testirani_podaci.flow_from_directory(
    testiran_direktorij,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 3)))
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
tensorboard = TensorBoard(log_dir='./logs')

povijest = model.fit(
    treniran_generator,
    epochs=50,
    validation_data=provjeravani_podaci,
    callbacks=[checkpoint, tensorboard]
)

model.load_weights('best_model.keras')

testirani_gubitak, testirani_preciznost = model.evaluate(testirani_generator)
print(f'Točnost testa: {testirani_preciznost:.4f}')

y_predikcija1 = model.predict(testirani_generator)
y_predikcija2 = np.argmax(y_predikcija1, axis=1)
y_istina = testirani_generator.classes

matrica_zabune = confusion_matrix(y_istina, y_predikcija2)
plt.figure(figsize=(12, 10))
sns.heatmap(matrica_zabune, annot=True, fmt="d", cmap='Blues')
plt.ylabel('tocan')
plt.xlabel('predviden')
plt.show()
