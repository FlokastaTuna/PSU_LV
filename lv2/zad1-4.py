import numpy as np
import matplotlib.pyplot as plt

#1.zad
kvadrat = np.array([[1,1], [3, 1], [3,2], [2,2], [1,1]])
plt.plot(kvadrat[:,0], kvadrat[:,1], color='red')
plt.plot(kvadrat[:,0], kvadrat[:,1], 'bo')
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('primjer')
plt.axis([0,4,0,4])
plt.show()

#2.zad
data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6),delimiter=",", skiprows=1)
plt.scatter(data[:,0], data[:,3], s=20, c='red', marker='s')
plt.scatter(data[:,0], data[:,5], s=data[:,5]*5, c='blue', marker='o')
print(data[:,5])
plt.xlabel('mpg')
plt.ylabel('hp(crvena), wt(plava)')
plt.title('potrosnja automobila')
if np.any(data[:,1]==6):
    uvjet=data[data[:, 1] == 6, 0]
    prosjek=np.mean(uvjet)
    prosjek=np.round(prosjek,2)
    minimum=min(uvjet)
    maks=max(uvjet)
print(prosjek, minimum, maks)
plt.show()

#3.zad
img = plt.imread("tiger.png")

plt.imshow(img + 0.2)
plt.title('veca svjetlina')
plt.show()


plt.imshow(np.rot90(img, k=-1))
plt.title('rotirana')
plt.show()


plt.imshow(np.fliplr(img))
plt.title('mirrorana')
plt.show()


plt.imshow(img[::10, ::10])
plt.title('10 puta manja rez')
plt.show()


visina, sirina, _ = img.shape
sirina_cetvrtine=sirina//4
cetvrtina = np.ones_like(img)
cetvrtina[:, sirina_cetvrtine:2*sirina_cetvrtine] = img[:, sirina_cetvrtine:2*sirina_cetvrtine]

plt.imshow(cetvrtina)
plt.title('cetvrtina slika')
plt.show()

#4.zad
def slikanje(kvadrati_velicina, kvadrati_visina, kvadrati_sirina):
    slika_visina = kvadrati_velicina * kvadrati_visina
    slika_sirina = kvadrati_velicina * kvadrati_sirina

   
    crno = np.zeros((slika_visina, slika_sirina))
    
    for i in range(kvadrati_visina):
        for j in range(kvadrati_sirina):
            if (i + j) % 2 != 0: 
                crno[i*kvadrati_velicina:(i+1)*kvadrati_velicina, j*kvadrati_velicina:(j+1)*kvadrati_velicina] = 255
    
    return crno

kvadrati_velicina = 50
kvadrati_visina = 4
kvadrati_sirina = 5

slika = slikanje(kvadrati_velicina, kvadrati_visina, kvadrati_sirina)

plt.imshow(slika, cmap='gray', vmin=0, vmax=255)
plt.show()