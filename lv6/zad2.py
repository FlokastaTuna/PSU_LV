from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def podaci(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        x, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        
    elif flagc == 2:
        random_state = 148
        x, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        x = np.dot(x, transformation)
        
    elif flagc == 3:
        random_state = 148
        x, y = datasets.make_blobs(n_samples=n_samples, centers=4, cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=random_state)

    elif flagc == 4:
        x, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
        
    elif flagc == 5:
        x, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    
    else:
        x = []
        
    return x

n_samples = 500
flagc = 5

x = podaci(n_samples, flagc)

inercija = []
klasteri = range(1, 21)

for n_clusters in klasteri:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(x)
    inercija.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(klasteri, inercija, marker='o')
plt.xlabel('klasteri')
plt.ylabel('iznos kriterijske funkcije')
plt.title('inercija za razlicite brojeve klastera')
plt.xticks(klasteri)
plt.grid(True)
plt.show()

#dobar broj klastera je onaj koji vidimo iz grafa kada se inercija uspori