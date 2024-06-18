import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

image_path = 'example_grayscale.png'
image = mpimg.imread(image_path)

plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.title('Originalna slika')
plt.axis('off')
plt.show()

X = image.reshape(-1, 1)

def kvantiziraj_sliku(X, n_clusters):
    k_means = KMeans(n_clusters=n_clusters, n_init=1)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    kompresana_slika = np.choose(labels, values)
    kompresana_slika.shape = image.shape
    return kompresana_slika

klasteri = [2, 5, 10, 20]
plt.figure(figsize=(12, 8))

for i, n_clusters in enumerate(klasteri, 1):
    kompresana_slika = kvantiziraj_sliku(X, n_clusters)
    plt.subplot(2, 2, i)
    plt.imshow(kompresana_slika, cmap='gray')
    plt.title(f'sa {n_clusters} klastera')
    plt.axis('off')

plt.tight_layout()
plt.show()

#kako povecavamo broj klastera slika pocinje liciti orginalu