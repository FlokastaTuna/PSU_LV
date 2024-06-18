import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

filename = 'E:\Opcenito stvari\Faks-stručni\2.god\4.semestar\PSU\Labosi\lv8\slika.png'


orginal_slika = mpimg.imread(filename)  
slika = color.rgb2gray(orginal_slika)
slika = resize(slika, (28, 28))


plt.imshow(slika, cmap=plt.get_cmap('gray'))
plt.axis('off')  
plt.show()


slika = slika.reshape(1, 28, 28, 1)
slika = slika.astype('float32') / 255.0


najbolji_model = models.load_model('best_model.keras')


predikcija = najbolji_model.predict(slika)
predikcija_klasa = np.argmax(predikcija)


print(f"Predicted class: {predikcija_klasa}")

#klasifikacija rotiranih slika ili slika koje nisu na pocetnom polozaju je teza