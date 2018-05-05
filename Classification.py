# Classification

from math import sqrt, factorial, isnan
import warnings
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from numba import guvectorize, vectorize, float64
import pandas as pd
import itertools as it
import timeit
import copy
import time

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def Derivace(data, krok, derivace):
    data = np.array(data)
    kon = len(data)-1
    derivace[0] = (data[1]-data[0])/krok
    derivace[kon] = (data[kon]-data[kon-1])/krok
    for i in range(1,len(data)-1):
            derivace[i] = (data[i+1]-data[i-1])/(2*krok)

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def Exp_Moving_Mean(data, window, emm):
    gamma = 0.9**np.arange(window)
    gamm = 0.9**np.arange(window)[::-1]
    count = 0
    dolni_index = 0
    for i in range(window):
        count+=1
        emm[i] = (sum(data[dolni_index: i + 1]*gamma[dolni_index: i + 1][::-1]))*(1/count)
    for i in range(window, len(data)):
        dolni_index = i + 1 - window
        emm[i] = (sum(data[dolni_index: i + 1]*gamm))*(1/count)

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def Moving_Mean(a, window, out):
    asum = 0.0
    count = 0
    for i in range(window):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window, len(a)):
        asum += a[i] - a[i - window]
        out[i] = asum / count

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def Moving_Variance(data, okno, rozptyl):
    mm = Moving_Mean(data, okno)
    dolni_index = 0
    count = 0
    for i in range(okno):
        count+=1
        rozptyl[i] = sum((data[dolni_index: i + 1] - mm[i])**2)*(1/count)
    for i in range(okno, len(data)):
        dolni_index = i + 1 - okno
        rozptyl[i] = sum((data[dolni_index: i + 1] - mm[i])**2)*(1/count)


def rozptyl_od_poc_fce(data, a_prumer_od_poc):
    odchylka = np.zeros(len(data))
    for i in range(len(data)):
        odchylka[i]= (1 / ( i + 1 )) * sum((data[0 : i + 1] - a_prumer_od_poc[0 : i + 1]) ** 2)
    return odchylka

def Srovnej(res, data, pocet_stavu = 3):
    # já vlastně přehazju data tak aby byla co největší schoda s res (skutečné výsledky)
    # když za data zadám stavy dostanu pak vektor stavů přerovnaný podle nám známého řešení(res)
    # to znamená že se bych je pak mohl přiradit správně do tříd, pokud bude známo řešení dopředu
    if(pocet_stavu == 2):
        [temp0, temp1] = np.unique(data)

        reverse = np.copy(data)
        reverse[reverse == temp0] = -1
        reverse[reverse == temp1] = temp0
        reverse[reverse == -1] = temp1

        right_sorted = [sum(res == data),sum(res == reverse)]
        vector = np.vstack((data,reverse))
        return([max(right_sorted), vector[np.argmax(right_sorted),:]])
    else:
        if len(np.unique(data)) == 2:
            Srovnej(res, data, 2)

        [temp0, temp1, temp2] = np.unique(data)
        perm = np.array([temp0, temp1, temp2])
        right_sorted = []
        vector = np.copy(data)
        for i in range(factorial(pocet_stavu) - 1):
            vector = np.vstack((vector, np.copy(data)))
        j = 0
        for p in multiset_permutations(perm):
            vector[j][vector[j] == temp0] = -1
            vector[j][vector[j] == temp1] = -2
            vector[j][vector[j] == temp2] = -3
            vector[j][vector[j] == -1] = p[0]
            vector[j][vector[j] == -2] = p[1]
            vector[j][vector[j] == -3] = p[2]
            right_sorted.append(sum(vector[j] == res))
            j+=1
        return([max(right_sorted), vector[np.argmax(right_sorted),:]])

def Accuracy(výsledek, stavy, pocet_stavu, srovnat = True):
    if srovnat:
        if len(výsledek)!=len(stavy):
            print("stavy a výsledky nesouhlasí dimenze")
            return
            #print(Confusion_Matrix(výsledek, stavy, pocet_stavu))
        if pocet_stavu <=2:
                #předpokládám že pří dvou stvech nebudu mít víc chyb než správných klasifikací
            součet = max(sum(výsledek == stavy),sum(výsledek != stavy))
            return [součet/len(stavy),len(stavy) - součet]
        else:
                přesnost = Srovnej(výsledek, stavy)[0]
                return [přesnost / len(výsledek),int(len(výsledek) - přesnost)]
    else:
        sou= sum(výsledek == stavy)
        return [sou / len(stavy), int(len(stavy) - sou)]

def Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat = True):
    if srovnat == True:
        srovnaný = Srovnej(výsledek, stavy, pocet_stavu)[1]
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
    #print(tabulka)
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
            #print("Precision was NaN ")
        if isnan(recall[k]):
            recall[k] = 0
            #print("Recall was NaN")
    #print(tabulka)
    return [precision, recall]


def Set_Features(data_set, šum = True, velikost_sumu = 1/40, delka_okna = 10, prvni_derivace = True,
               druha_derivace = True, suma_zleva = False, aritmeticky_prumer = False,
               rozptyl = False, vypis_nastacene_vlastnosti = False):

    if vypis_nastacene_vlastnosti == True:
        print('Délka okna:', delka_okna,
              '\n Šum:', šum,
              '\n Velikost šumu: ', velikost_sumu,
              '\n Prvni_derivace:',prvni_derivace,
              '\n Druha_derivace:', druha_derivace,
              '\n Suma_zleva:', suma_zleva,
              '\n Aritmeticky_prumer:', aritmeticky_prumer,
              '\n Rozptyl:', rozptyl)

    X = np.array(data_set)
    vlastnosti = np.zeros(6).tolist()

    if šum == True:
        šum = np.random.randn(len(data_set))
        X = X+šum*velikost_sumu

    XX=X
    # samotné XX je jen data_set
    # do X jsou odteď přidané i feature

    if prvni_derivace == True:
        Dx1 = Derivace(XX,1)
        X = np.vstack([X, Dx1])
        vlastnosti[0] = Dx1

    if druha_derivace == True:
        Dx1 = Derivace(XX,1)
        Dx2 = Derivace(Dx1,1)
        X = np.vstack([X, Dx2])
        vlastnosti[1] = Dx2

    if suma_zleva == True:
        Suma_L = Exp_Moving_Mean(data, delka_okna)
        #[suma_zleva_fce(XX, x, delka_okna) for x in range(len(XX))]
        X = np.vstack([X, Suma_L])
        vlastnosti[2] = Suma_L

    if aritmeticky_prumer == True:
        #Arit_Pr = [aritmeticky_prumer_fce(XX, x, delka_okna) for x in range(len(XX))]
        Arit_pr = Moving_Mean(XX, delka_okna)
        X = np.vstack([X, Arit_Pr])
        vlastnosti[3] = Arit_Pr

    if rozptyl == True:
        Rozptyl = Moving_Variance(XX, delka_okna)
        X = np.vstack([X, Rozptyl])
        vlastnosti[4] = Rozptyl

    # transponuju teď už matici původního data setu a features
    return (X.T, XX, vlastnosti)


###########################################################################################################################################

def klasifikuj(data_set, kontolni_data_set, pocet_stavu = 2, šum = True,
               velikost_sumu = 1/40, delka_okna = 10, prvni_derivace = True,
               druha_derivace = False, suma_zleva = False, aritmeticky_prumer = False,
               rozptyl = False, kresli_a_piš = True):

    [X, XX, vlastnosti] = Set_Features(data_set, šum, velikost_sumu, delka_okna, prvni_derivace,
               druha_derivace, suma_zleva, aritmeticky_prumer, rozptyl, kresli_a_piš)

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

        if rozptyl == True:
            plt.plot(vlastnosti[4])

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
    means = np.zeros((2**5 -1,2))
    # střední hodnoty jsou 2D matice, kde první sloupec je přesnost a druhý je počet chyb
    combinace = []
    # kombinace
    iterace = 0

    time=np.zeros(2**5 - 1)
    # vektor časů průběhu kombinací

    for combin in it.product([0,1],repeat=5):
        # repeat je počet možných feature, který lze použít
        start = timeit.default_timer()

        if combin == (0,0,0,0,0):
             # kombinace feature (0,0,0,0,0) je k ničemu, stejně jí HMM nepřechroustá a spadne
            continue

        combinace.append(combin)

        temp = klasifikuj(D, řešení, počet_stavů, šum, velikost_sumu, delka_okna,
                          combin[0], combin[1], combin[2], combin[3], combin[4], False)

        for i in range(počet_opakování-1 ):
            temp = np.vstack((temp, klasifikuj(D, řešení, počet_stavů, šum, velikost_sumu, delka_okna,
                                               combin[0], combin[1], combin[2], combin[3], combin[4], False)))

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
#################################################################


def validuj(model, train_data, test_data, delka_okna =[], parametry  = [], unsupervised = True):
    warnings.filterwarnings('ignore')
    if not unsupervised:
        labels = train_data[0][2]
        for lab in test_data[1:]:
            labels = h.stack((labels,lab[2]))
        labels = labels.T
    Labely = test_data[0][2]
    for lab in test_data[1:]:
        Labely = np.hstack((Labely, lab[0][2]))
    if parametry:
        if not delka_okna:
            delka_okna = 10
        if len(parametry) < 5:
            parametry = parametry + np.zeros(5-len(parametry)).tolist()

        training_data = Set_Features(train_data[0][1], False, 0, delka_okna, parametry[0], parametry[1],parametry[2],
                                     parametry[3], parametry[4])[0]
        for train in train_data[1:]:
            training_data = np.vstack((training_data, Set_Features(train[1], False, 0, delka_okna, parametry[0], parametry[1],
                                                                   parametry[2], parametry[3], parametry[4])[0]))
        testing_data = Set_Features(test_data[0][1], False, 0, delka_okna, parametry[0], parametry[1], parametry[2],
                                    parametry[3], parametry[4])[0]
        for test in test_data[1:]:
            testing_data = np.vstack((testing_data[0][1],Set_Features(test, False, 0, delka_okna, parametry[0], parametry[1], parametry[2],
                                        parametry[3], parametry[4])[0]))

        CLF = copy.copy(model)
        if unsupervised:
            stf = time.time()
            CLF.fit(training_data)
            endf = time.time()
        else:
            CLF.fit(training_data, labels)
        stp = time.time()
        states = CLF.predict(testing_data)
        endp = time.time()

        [acc, mis] = Accuracy(Labely, states, 3, unsupervised)
        [f, fa] = F_Measure(Labely, states, 3, unsupervised)
        [p, r] = Precision_n_Recall(Labely,states,3, unsupervised)
        panda = list(zip([tuple(parametry),0], [delka_okna,0], [acc,0], [mis,0], [f[0],0], [f[1],0],
                                                [f[2],0], [fa,0], [p[0],0], [p[1],0], [p[2],0], [r[0],0], [r[1],0], [r[2],0]))
        dpanda = pd.DataFrame(data = panda, columns = ['Kombinace rysů','délka úseku', 'Accuracy', 'Chyby',
                                                    'F míra stavu 0', 'F míra stavu 1', 'F míra stavu 2',
                                                    'F míra průměrná', 'Precision stavu 0','Precision stavu 1',
                                                    'Precision stavu 2', 'Recall stavu 0', 'Recall stavu 1',
                                                    'Recall stavu 2'])
        del CLF, training_data, testing_data, f, fa, acc, mis, p, r #, states
        return dpanda, states, endf-stf, endp-stp
    else:
        [combinace, accuracy, chyby, F0, F1, F2, F_average, P0, P1, P2, R0, R1, R2, okno] = [
                                                    [],[],[],[],[],[],[],[],[],[],[],[],[],[]]

        if not delka_okna:
            delka_okna = [10]
        for okna in delka_okna:
            iterace = 0

            for combin in it.product([0,1],repeat=5):
                # repeat je počet možných feature, který lze použít
                if combin == (0,0,0,0,0):
                    continue
                combinace.append(combin)
                okno.append(okna)

                training_data = Set_Features(train_data[0][1], False, 0, okna,combin[0], combin[1], combin[2], combin[3], combin[4])[0]
                for train in train_data[1:]:
                    training_data = np.vstack((training_data,Set_Features(train[1], False, 0, okna, combin[0], combin[1], combin[2], combin[3], combin[4])[0]))

                testing_data = Set_Features(test_data[0][1], False, 0, okna, combin[0],combin[1], combin[2], combin[3], combin[4])[0]
                for test in test_data[1:]:
                    testinging_data = np.vstack((testing_data,Set_Features(test[0][1], False, 0, okna,combin[0], combin[1], combin[2], combin[3], combin[4])[0]))

                CLF = copy.copy(model)
                if unsupervised:
                    CLF.fit(training_data)
                else:
                    CLF.fit(training_data, labels)

                states = CLF.predict(testing_data)
                print(states)

                [acc, mis] = Accuracy(Labely, states, 3, unsupervised)
                accuracy.append(acc), chyby.append(mis)
                [f, fa] = F_Measure(Labely, states, 3, unsupervised)
                F0.append(f[0]), F1.append(f[1]), F2.append(f[2]), F_average.append(fa)
                [p, r] = Precision_n_Recall(Labely,states,3, unsupervised)
                P0.append(p[0]), P1.append(p[1]), P2.append(p[2])
                R0.append(r[0]), R1.append(r[1]), R2.append(r[2])
                del CLF, training_data, testing_data, f, fa, acc, mis, p, r, states
                iterace +=1
                print(iterace)

        panda = list(zip(combinace, okno, accuracy, chyby, F0, F1, F2, F_average, P0, P1, P2, R0, R1, R2))
        dpanda = pd.DataFrame(data = panda, columns = ['Kombinace rysů','délka úseku', 'Accuracy', 'Chyby',
                                                    'F míra stavu 0', 'F míra stavu 1', 'F míra stavu 2',
                                                    'F míra průměrná', 'Precision stavu 0','Precision stavu 1',
                                                    'Precision stavu 2', 'Recall stavu 0', 'Recall stavu 1',
                                                    'Recall stavu 2'])

        return dpanda
