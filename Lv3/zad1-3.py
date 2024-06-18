import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pandas as pd 

import urllib
import urllib.request
import xml.etree.ElementTree as ET

mtcars = pd.read_csv('mtcars.csv')

#1.zad
print("--------------------------------------------")
print("1. Kojih 5 automobila ima najveću potrošnju?")
print(mtcars.sort_values(by=['mpg'], ascending = True).head(5))

print("--------------------------------------------------------------")
print("2. Koja tri automobila s 8 cilindara imaju najmanju potrošnju?")
osam_cilindra = mtcars[mtcars.cyl == 8]
print(osam_cilindra.sort_values(by=['mpg'], ascending = True).tail(3))

print("---------------------------------------------------------")
print("3. Kolika je srednja potrošnja automobila sa 6 cilindara?")
sest_cilindra = mtcars[mtcars.cyl == 6] 
sest_cilindra_potrosnja = sest_cilindra["mpg"].mean()
print(sest_cilindra_potrosnja)

print("-----------------------------------------------------------------------------------")
print("4. Kolika je srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs?")
cetiri_cilindra_masa = mtcars[(mtcars.cyl == 4) & (mtcars.wt>2) & (mtcars.wt<2.2)]
cetiriclindra_potrosnja = cetiri_cilindra_masa["mpg"].mean()
print(cetiriclindra_potrosnja)

print("-----------------------------------------------------------------------------------------")
print("5. Koliko je automobila s ručnim, a koliko s automatskim mjenjačem u ovom skupu podataka?")
mjenjac = mtcars.groupby('am').car
print(mjenjac.count())

print("----------------------------------------------------------------------------------")
print("6. Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga?")
automatik = mtcars[(mtcars.am == 0) & (mtcars.hp>100)]
print(automatik["car"].count())

print("--------------------------------------------------")
print("7. Kolika je masa svakog automobila u kilogramima?")
tezina = 0.45359  
CarIndex = 0
for wt in mtcars["wt"]:   
    iznos = (wt * 1000) * tezina         
    print(str(CarIndex) + ". " + mtcars.car[CarIndex] + ": %.3f kg" % (iznos))
    CarIndex = CarIndex + 1 


#2.zad

# 2.1. Pomoću barplot-a prikažite na istoj slici potrošnju automobila s 4, 6 i 8 cilindara.
cilidnri_cetiri = mtcars[mtcars.cyl == 4]
cilindri_sest = mtcars[mtcars.cyl == 6]
cilindri_osam = mtcars[mtcars.cyl == 8]

cetiri_indeksi = np.arange(0, len(cilidnri_cetiri), 1)
sest_indeksi = np.arange(0, len(cilindri_sest), 1) 
osam_indeksi = np.arange(0, len(cilindri_osam), 1) 

widthBetweenColumns = 0.3

plt.figure()

plt.bar(cetiri_indeksi, cilidnri_cetiri["mpg"], widthBetweenColumns, color="r")
plt.bar(sest_indeksi + widthBetweenColumns, cilindri_sest["mpg"], widthBetweenColumns, color="g")
plt.bar(osam_indeksi + 2*widthBetweenColumns, cilindri_osam["mpg"], widthBetweenColumns, color="b")
plt.title("Potrosnja automobila s 4, 6 i 8 cilindara")
plt.xlabel('auto')
plt.ylabel('mpg')
plt.legend(['4 cilindra','6 cilindara','8 cilindara'], loc=1)
plt.grid(axis='y', linestyle='--')
plt.show()

# 2.2. Pomoću boxplot-a prikažite na istoj slici distribuciju težine automobila s 4, 6 i 8 cilindara.
cetiri_tezina=[]
sest_tezina=[]
osam_tezina=[]

for i in cilidnri_cetiri["wt"]:
     cetiri_tezina.append(i) 

for i in cilindri_sest["wt"]:
     sest_tezina.append(i)  

for i in cilindri_osam["wt"]:
     osam_tezina.append(i)  

plt.figure()
plt.boxplot([cetiri_tezina, sest_tezina, osam_tezina], positions = [4,6,8]) 
plt.title("Tezina automobila s 4, 6 i 8 cilindara")
plt.xlabel('Broj klipova')
plt.ylabel('Tezina wt')
plt.grid(axis='y',linestyle='--')
plt.show()

# 2.3. Pomoću odgovarajućeg grafa pokušajte odgovoriti na pitanje imaju li automobili s ručnim mjenjačem veću
# potrošnju od automobila s automatskim mjenjačem?

automatici = mtcars[(mtcars.am == 0)]
automatici_potrosnja=[]
for i in automatici["mpg"]:
     automatici_potrosnja.append(i)
    
manualCars = mtcars[(mtcars.am == 1)]
CarConsumption_manual=[]
for i in manualCars["mpg"]:
     CarConsumption_manual.append(i)

plt.figure()
plt.boxplot([CarConsumption_manual, automatici_potrosnja], positions = [0,1])
plt.title("Usporedba potrosnja automobila s rucnim, odnosno automatskim mjenjacem")
plt.ylabel('miles per gallon')
plt.xlabel('automatski mjenjac                         rucni mjenjac')
plt.grid(axis='y',linestyle='--')
plt.show()

# 2.4. Prikažite na istoj slici odnos ubrzanja i snage automobila za automobile s ručnim odnosno automatskim
# mjenjačem. 
automatici_ubrzanje=[]
for i in automatici["qsec"]:  
     automatici_ubrzanje.append(i)
    
automatici_snaga=[]  
for i in automatici["hp"]:  
     automatici_snaga.append(i)
    
manualci_ubrzanje=[]    
for i in manualCars["qsec"]:  
     manualci_ubrzanje.append(i)
    
manualci_snaga=[]    
for i in manualCars["hp"]:  
     manualci_snaga.append(i)

plt.figure()
plt.scatter(automatici_ubrzanje, automatici_snaga, marker='+')  
plt.scatter(manualci_ubrzanje, manualci_snaga, marker='^', facecolors='none', edgecolors='r')
plt.title("Odnos ubrzanja i snage automobila s rucnim u odnosu na automatski mjenjac")
plt.ylabel('Snaga - hp')
plt.xlabel('Ubrzanje - qsec')
plt.legend(["Automatski mjenjac","Rucni mjenjac"])
plt.grid(linestyle='--')
plt.show()


#3.zad
# 3.1. Dohvaćanje mjerenja dnevne koncentracije lebdećih čestica PM10 za 2017. godinu za grad Osijek.
url = "http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=5&vrijemeOd=02.01.2017&vrijemeDo=01.01.2018"

kvaliteta = urllib.request.urlopen(url).read()
root = ET.fromstring(kvaliteta)
print(root)

df = pd.DataFrame(columns=('mjerenje', 'vrijeme'))

i = 0
while True:
    
    try:
         obj = root.getchildren()[i].getchildren()
    except:
         break
    
    row = dict(zip(['mjerenje', 'vrijeme'], [obj[0].text, obj[2].text]))
    row_s = pd.Series(row)
    row_s.name = i
    df = df.append(row_s)
    df.mjerenje[i] = float(df.mjerenje[i])
    i = i + 1

df.vrijeme = pd.to_datetime(df.vrijeme, utc=True)


df['month'] = pd.DatetimeIndex(df['vrijeme']).month 
df['dayOfweek'] = pd.DatetimeIndex(df['vrijeme']).dayofweek

 # 3.2. Ispis tri datuma u godini kada je koncentracija PM10 bila najveća.
topPM10values = df.sort_values(by=['mjerenje'], ascending=False)
print("\nTri datuma kad je koncentracija PM10 u 2017. bila najveca: ")
print(topPM10values['vrijeme'].head(3))