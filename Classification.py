# Classification

from math import sqrt, factorial, isnan
import warnings
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
import timeit

def Derivace(data, krok = 1):
    data = np.array(data)
    derivace = np.zeros(len(data))
    for i in range(len(data)):
        if i == 0:
            derivace[i] = (data[1]-data[0])/krok
        elif i == len(data)-1:
            derivace[i] = (data[i]-data[i-1])/krok
        else:
            derivace[i] = (data[i+1]-data[i-1])/(2*krok)
    return derivace

def suma_zleva_fce(data, pozice, okno = 10):
    pozice += 1
    Y = np.zeros(len(data))
    temp_generator = [0.9 ** x for x in range(pozice)]

    if pozice-okno < 0:
        u = 0
    else:
            u = pozice - okno
    for i in range(u, pozice):
        Y[i] = temp_generator[pozice - 1 - i] * data[i]

    if pozice >= okno:
        delitel = okno
    else:
        delitel = pozice

    return sum(Y) / delitel


def aritmeticky_prumer_fce(data, pozice, okno = 10):
    pozice += 1
    Y = np.zeros(len(data))
    if pozice - okno < 0:
        u = 0
        delitel = pozice
    else:
            u = pozice - okno
            delitel = okno

    for i in range(u, pozice):
        Y[i] = Y[i - 1] + data[i]

    return Y[pozice-1] / delitel


def rozptyl_fce(data, okno = 10):
    Odchylka = np.zeros(len(data))
    Aritm = [aritmeticky_prumer_fce(data, x, okno) for x in range(len(data))]

    for i in range(len(data)):
        if i + 1 - okno < 0:
            dolni_index = 0;
            delitel = i + 1
        else:
            dolni_index = i + 1 - okno
            delitel = okno

        Odchylka[i] = sqrt((1 / delitel) * sum((data[dolni_index: i + 1] - Aritm[dolni_index: i + 1]) ** 2))

    return Odchylka


def rozptyl_od_poc_fce(data, a_prumer_od_poc):
    odchylka = np.zeros(len(data))
    for i in range(len(data)):
        odchylka[i]= sqrt((1 / ( i + 1 )) * sum((data[0 : i + 1] - a_prumer_od_poc[0 : i + 1]) ** 2))
    return odchylka

def srovnej(res, data, pocet_stavu = 3):
    # já vlastně přehazju data tak aby byla co největší schoda s res (skutečné výsledky)
    # když za data zadám stavy dostanu pak vektor stavů přerovnaný podle nám známého řešení(res)
    # to znamená že se bych je pak mohl přiradit správně do tříd, pokud bude známo řešení dopředu

    if(pocet_stavu == 2):
        temp0 = data[0]
        i = 0
        while temp0 == data[i]:
            i += 1;
        temp1 = data[i]

        reverse = np.copy(data)
        for i in range(len(data)):
            if reverse[i] == temp0:
                reverse[i] = -1
            if reverse[i] == temp1:
                reverse[i] = -2

        for i in range(len(data)):
            if reverse[i] == -1:
                reverse[i] = temp1
            if reverse[i] == -2:
                reverse[i] = temp0

        vektor_přesností = [sum(res == data),sum(res == reverse)]
        v = np.vstack((data,reverse))
        return([max(vektor_přesností),v[np.argmax(vektor_přesností),:]])
    else:
        temp0 = data[0]
        i = 0
        while temp0 == data[i]:
            i += 1;
        temp1 = data[i]
        while (data[i] == temp1 or data[i] == temp0):
            i += 1
        temp2 = data[i]
        perm = np.array([temp0, temp1, temp2])

        vektor_přesností = []

        v = np.copy(data)
        for i in range(factorial(pocet_stavu) - 1):
            v = np.vstack((v, np.copy(data)))
        j = 0
        for p in multiset_permutations(perm):
            for i in range(len(v[j])):
                if v[j,i] == temp0:
                    v[j,i] = -1
                if v[j,i] == temp1:
                    v[j,i] = -2
                if v[j,i] == temp2:
                    v[j,i] = -3

            [t0, t1 , t2] = p

            for i in range(len(v[j])):
                if v[j,i] == -1:
                    v[j,i] = t0
                if v[j,i] == -2:
                    v[j,i] = t1
                if v[j,i] == -3:
                    v[j,i] = t2

            součet = sum(v[j] == res)

            vektor_přesností.append(součet)
            j+=1
        return([max(vektor_přesností), v[np.argmax(vektor_přesností),:]])


def Accuracy(výsledek, stavy, pocet_stavu):
    if len(výsledek)!=len(stavy):
        print("stavy a výsledky nesouhlasí dimenze")
        return
    #print(Confusion_Matrix(výsledek, stavy, pocet_stavu))
    if pocet_stavu <=2:
        #předpokládám že pří dvou stvech nebudu mít víc chyb než správných klasifikací
        součet = max(sum(výsledek == stavy),sum(výsledek != stavy))
        return np.array([součet/len(stavy),len(stavy)-součet])
    else:
        přesnost = srovnej(výsledek, stavy)[0]
        return np.array([přesnost / len(výsledek),int(len(výsledek) - přesnost)])

def Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat = True):
    if srovnat == True:
        srovnaný = srovnej(výsledek, stavy, pocet_stavu)[1]
        #print(srovnaný)
        #print(výsledek)
    else:
        srovnaný = stavy
    tabulka = np.zeros((pocet_stavu,pocet_stavu), dtype = 'int64')
    for i in range(pocet_stavu):
        for j in range(pocet_stavu):
            tabulka[i,j] = sum(výsledek[k] == i and srovnaný[k] == j for k in range(len(výsledek)))
    return tabulka

def F_Measure(výsledek, stavy, pocet_stavu, srovnat = True):
    tabulka = Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat)
    FM = np.zeros(pocet_stavu)
    for k in range(pocet_stavu):
        FM[k] = 2 * tabulka[k,k] / (sum(tabulka[k,:]) + sum(tabulka[:,k]))
    # vracím [FM vektor, FM macro]
    # FM je F1_score('none') a suma je F1_score("macro") což je průměr F1 z všech tříd
    print(tabulka)
    return [FM, sum(FM) / pocet_stavu]

def Precision_n_Recall(výsledek, stavy, pocet_stavu, srovnat = True):
    tabulka = Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat)
    precision = np.zeros(pocet_stavu)
    recall = np.zeros(pocet_stavu)
    for k in range(pocet_stavu):
        precision[k] = tabulka[k,k] / sum(tabulka[:,k])
        recall[k] = tabulka[k,k] / sum(tabulka[k,:])
        if isnan(precision[k]):
            precision[k] = 0
            print("Precision was NaN ")
        if isnan(recall[k]):
            recall[k] = 0
            print("Recall was NaN")
    print(tabulka)
    return [precision, recall]


def Set_Features(data_set, šum = True, velikost_sumu = 1/40, delka_okna = 10, prvni_derivace = True,
               druha_derivace = True, suma_zleva = False, aritmeticky_prumer = False,
               rozptyl = False, rozptyl_od_pocatku = False, vypis_nastacene_vlastnosti = False):

    if vypis_nastacene_vlastnosti == True:
        print('Délka okna:', delka_okna,
              '\n Šum:', šum,
              '\n Velikost šumu: ', velikost_sumu,
              '\n Prvni_derivace:',prvni_derivace,
              '\n Druha_derivace:', druha_derivace,
              '\n Suma_zleva:', suma_zleva,
              '\n Aritmeticky_prumer:', aritmeticky_prumer,
              '\n Rozptyl:', rozptyl,
              '\n Rozptyl_od_pocatku:', rozptyl_od_pocatku)

    X = np.array(data_set)
    vlastnosti = np.zeros(6).tolist()

    if šum == True:
        šum = np.random.randn(len(data_set))
        X = X+šum*velikost_sumu

    XX=X
    # samotné XX je jen data_set
    # do X jsou odteď přidané i feature

    if prvni_derivace == True:
        Dx1 = Derivace(XX)
        X = np.vstack([X, Dx1])
        vlastnosti[0] = Dx1

    if druha_derivace == True:
        Dx1 = Derivace(XX)
        Dx2 = Derivace(Dx1)
        X = np.vstack([X, Dx2])
        vlastnosti[1] = Dx2

    if suma_zleva == True:
        Suma_L = [suma_zleva_fce(XX, x, delka_okna) for x in range(len(XX))]
        X = np.vstack([X, Suma_L])
        vlastnosti[2] = Suma_L

    if aritmeticky_prumer == True:
        Arit_Pr = [aritmeticky_prumer_fce(XX, x, delka_okna) for x in range(len(XX))]
        X = np.vstack([X, Arit_Pr])
        vlastnosti[3] = Arit_Pr

    if rozptyl == True:
        Rozptyl = rozptyl_fce(XX, delka_okna)
        X = np.vstack([X, Rozptyl])
        vlastnosti[4] = Rozptyl

    if rozptyl_od_pocatku == True:
        aritmetiky_prumer_od_pocatku = np.cumsum(XX) / (np.arange(len(XX)) + 1)
        Rozpt = rozptyl_od_poc_fce(XX, aritmetiky_prumer_od_pocatku)
        X = np.vstack([X, Rozpt])
        vlastnosti[5] = Rozpt

    # transponuju teď už matici původního data setu a features
    return (X.T, XX, vlastnosti)


###########################################################################################################################################

def klasifikuj(data_set, kontolni_data_set, pocet_stavu = 2, šum = True,
               velikost_sumu = 1/40, delka_okna = 10, prvni_derivace = True,
               druha_derivace = False, suma_zleva = False, aritmeticky_prumer = False,
               rozptyl = False, rozptyl_od_pocatku = False, kresli_a_piš = True):

    [X, XX, vlastnosti] = Set_Features(data_set, šum, velikost_sumu, delka_okna, prvni_derivace,
               druha_derivace, suma_zleva, aritmeticky_prumer, rozptyl, rozptyl_od_pocatku,
               kresli_a_piš)

    warnings.filterwarnings('ignore')

    HMM_klasifikace = GaussianHMM(pocet_stavu)

    HMM_klasifikace.fit(X)

    states = HMM_klasifikace.predict(X)

    if kresli_a_piš == True:
        T_label = np.arange(len(XX))

        plt.figure("Testovací data")
        plt.plot(T_label, XX, color = 'red')
        plt.show()

        print(states)

        plt.figure('Unsupervised hmm. %i stavů ' %pocet_stavu)
        plt.plot(T_label, XX, linewidth = 0.5)
        plt.scatter(T_label, XX, c = states, cmap = plt.cm.plasma)

        if suma_zleva == True:
            plt.plot(np.array(vlastnosti[2])+np.amin(data_set) / 2)

        if aritmeticky_prumer == True:
            plt.plot(vlastnosti[3])

        if rozptyl_od_pocatku == True:
            plt.plot(vlastnosti[4])

        if rozptyl == True:
            plt.plot(vlastnosti[5])

        plt.show()

        print("Vracím: přesnost, počet chyb, FM, průměr FM ze všech tříd, preciznost, \"recall\"")

    return [Accuracy(kontolni_data_set,states,pocet_stavu),
            F_Measure(kontolni_data_set,states,pocet_stavu),
            Precision_n_Recall(kontolni_data_set,states,pocet_stavu)]

###########################################################################################################################################

def přesnost_klasifikace(D, řešení, šum, velikost_sumu, počet_stavů, delka_okna, počet_opakování = 500, výpis = True):
    N = []
    # hodnoty z každé klasifikace ([úspěšnost, chyby])
    dN = []
    # tabulka hodnot v pandas
    means = np.zeros((63,2))
    # střední hodnoty jsou 2D matice, kde první sloupec je přesnost a druhý je počet chyb
    combinace = []
    # kombinace
    iterace = 0

    time=np.zeros(63)
    # vektor časů průběhu kombinací

    for combin in it.product([0,1],repeat=6):
        # repeat je počet možných feature, který lze použít
        start = timeit.default_timer()

        if combin == (0,0,0,0,0,0):
             # kombinace feature (0,0,0,0,0,0) je k ničemu, stejně jí HMM nepřechroustá a spadne
            continue

        combinace.append(combin)

        temp = klasifikuj(D, řešení, počet_stavů, šum, velikost_sumu, delka_okna,
                          combin[0], combin[1], combin[2], combin[3], combin[4], combin[5], False)

        for i in range(počet_opakování-1 ):
            temp = np.vstack((temp, klasifikuj(D, řešení, počet_stavů, šum, velikost_sumu, delka_okna,
                                               combin[0], combin[1], combin[2], combin[3], combin[4], combin[5], False)))

        N.append(temp)

        dtemp = (pd.DataFrame(data = temp, columns=['Procenta', 'Chyby']))

        dN.append(dtemp)

        means[iterace,0] = dtemp['Procenta'].mean()
        means[iterace,1] = dtemp['Chyby'].mean()

        if výpis == True:
            print(iterace)

        stop = timeit.default_timer()
        time[iterace] = stop-start

        iterace = iterace + 1

    return [N, dN, means, combinace, time]
