import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

image_path = 'example.png'
image = mpimg.imread(image_path)

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title('Originalna slika')
plt.axis('off')
plt.show()

v, s, d = image.shape
polje_slika = np.reshape(image, (v * s, d))

n_colors = 10
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(polje_slika)
labela = kmeans.predict(polje_slika)

kvantizirana_slika = kmeans.cluster_centers_[labela]
kvantizirana_slika = np.reshape(kvantizirana_slika, (v, s, d))

plt.figure(figsize=(8, 8))
plt.imshow(kvantizirana_slika)
plt.title(f'slika s {n_colors} boja')
plt.axis('off')
plt.show()
