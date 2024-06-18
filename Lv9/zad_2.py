import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os

model = load_model('best_model.keras')

def ucitaj_sliku(mjesto_slike):
    slika = image.load_img(mjesto_slike, target_size=(48, 48))
    slika_polje = image.img_to_array(slika)
    slika_polje = np.expand_dims(slika_polje, axis=0)
    slika_polje /= 255.0  
    return slika_polje

def uzmi_klase(train_dir):
    class_labels = sorted(os.listdir(train_dir))
    return class_labels

train_dir = 'E:\Opcenito stvari\Faks-stručni\2.god\4.semestar\PSU\Labosi\lv9\Train'

labele_klase = uzmi_klase(train_dir)

mjesto_slike = 'E:\Opcenito stvari\Faks-stručni\2.god\4.semestar\PSU\Labosi\lv9\znak.png' 

slika_polje = ucitaj_sliku(mjesto_slike)

predikcija = model.predict(slika_polje)
predikcija_klasa = np.argmax(predikcija, axis=1)[0]

print(f'Predicted class: {labele_klase[predikcija_klasa]}')

slika = image.load_img(mjesto_slike)
plt.imshow(slika)
plt.title(f'predikcija {labele_klase[predikcija_klasa]}')
plt.axis('promasaj')
plt.show()
