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
        x = np.dot(X, transformation)
        
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

def crtanje(x, kmeans):
    y_kmeans = kmeans.predict(X)
    plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
    plt.show()

n_samples = 500
flagc = 5 

x = podaci(n_samples, flagc)

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

crtanje(x, kmeans)

#promjenom na koji nacin se generiranaju podaci mijenja se rasprsenje podataka i oblik rasprsivanja