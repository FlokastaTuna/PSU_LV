#1.zad
def euro(sati, euro_h):
    total=sati*euro_h
    return(total)

sati=float(input("Radni sati: "))
euro_h=float(input("Eura po sati: "))
ukupno=euro(sati,euro_h)
print("ukupno ", ukupno)

#2.zad
def kategorija(ocjena):
    if ocjena>= 0.9 and ocjena <=1.0:
        return "A"
    elif ocjena >= 0.8:
        return "B"
    elif ocjena >= 0.7:
        return "C"
    elif ocjena >= 0.6:
        return "D"
    elif ocjena >= 0.0 and ocjena < 0.6:
        return "F"
    else:
        return "netocan unos"

ocjena=-1.0
while(ocjena < 0.0 or ocjena > 1.0):
    ocjena=float(input("Unesite ocijenu: "))
    if ocjena < 0.0 or ocjena > 1.0:
        print("ocjena mora bit između 0 i 1, ponovi unos")
       
kat=kategorija(ocjena)
print("Kategorija ocijene je: ", kat)

#3.zad
def operacija(brojevi):
    duzina=len(brojevi)
    print("Brojevi koje ste unijeli:", brojevi)
    print("Broj unesenih brojeva: ",duzina)
    print("Srednja vrijednost brojeva:", sum(brojevi)/duzina)
    print("Minimalna vrijednost brojeva: ", min(brojevi))
    print("Maksimalna vrijednost brojeva: ", max(brojevi))
   

brojevi=[]
duzina=0
ulaz=0
while(True):
    ulaz=input("Unesite broj ili Done za kraj unosa:")
    if(ulaz=="Done"):
        break
    broj=float(ulaz)
    brojevi.append(broj)
operacija(brojevi)

#4.zad
ime_datoteke = input("Ime datoteke: ")
dat_dir = "E:\Opcenito stvari\Faks-stručni\2.god\4.semestar\PSU\Labosi\lv1"+ime_datoteke

try:
    datoteka = open(dat_dir, 'r')
except:
    print("Datoteka ne postoji!")
suma = 0
brojac = 0

for line in datoteka:
    line = line.rsplit()
    if ("X-DSPAM-Confidence:" in line):
        brojac += 1
        suma += float(line[1])

print("Average X-DSPAM-Confidence: ", suma/brojac)

datoteka.close()

#5.zad
dat = open("E:\Opcenito stvari\Faks-stručni\2.god\4.semestar\PSU\Labosi\lv1\songs.txt", 'r')


rijecnik = {}


for line in dat:
    line = line.rsplit()
    for rijec in line:
        if rijec in rijecnik:
            rijecnik[rijec] += 1
        else:
            rijecnik[rijec] = 1


unikatne_rijeci = []


for rijec in rijecnik:
    if (rijecnik[rijec] == 1):
        unikatne_rijeci.append(rijec)

print(rijecnik)


print("broj jedinstvenih rijeci ", len(unikatne_rijeci))

print(unikatne_rijeci)

#6.zad
def ucitaj_sms(ime_dat):
    ham_poruka = []
    spam_poruka = []
    
    try:
        with open(ime_dat, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split('\t')
                label = parts[0]
                message = parts[1].strip()
                
                if label == 'ham':
                    ham_poruka.append(message)
                elif label == 'spam':
                    spam_poruka.append(message)
    
    except FileNotFoundError:
        print("nema datoteke")
        return None
    
    return ham_poruka, spam_poruka

def broj_rj(poruka):
    sve_rijeci = sum(len(message.split()) for message in poruka)
    br_poruka = len(poruka)
    
    if br_poruka == 0:
        return 0
    
    return sve_rijeci / br_poruka

def broji_poruke(poruka):
    count = 0
    for message in poruka:
        if message.endswith('!'):
            count += 1
    return count

def main():
    ime_dat = "SMSSpamCollection.txt" 
    
    ham_poruka, spam_poruka = ucitaj_sms(ime_dat)
    
    if ham_poruka is not None and spam_poruka is not None:
        prosjek_ham = broj_rj(ham_poruka)
        prosjek_spam = broj_rj(spam_poruka)
        
        print("Prosječan broj riječi u porukama koje su tipa ham:", prosjek_ham)
        print("Prosječan broj riječi u porukama koje su tipa spam:", prosjek_spam)
        
        spamsa_esklamacijom = broji_poruke(spam_poruka)
        print("Broj SMS poruka koje su tipa spam i završavaju uskličnikom:", spamsa_esklamacijom)

if __name__ == "__main__":
    main()
