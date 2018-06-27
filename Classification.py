# Classification

from math import factorial, isnan
import warnings
import time
import itertools as it
import numpy as np
from copy import copy
from sympy.utilities.iterables import multiset_permutations
from hmmlearn.hmm import GaussianHMM
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs
from numba import guvectorize
import pandas as pd
import progressbar

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def derivace(data, krok, deriv):
    data = np.array(data)
    kon = len(data)-1
    deriv[0] = (data[1]-data[0])/krok
    deriv[kon] = (data[kon]-data[kon-1])/krok
    for i in range(1, len(data)-1):
        deriv[i] = (data[i+1]-data[i-1])/(2*krok)

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def exp_moving_mean(data, window, emm):
    gamma = 0.9**np.arange(window)
    gamm = 0.9**np.arange(window)[::-1]
    count = 0
    dolni_index = 0
    for i in range(window):
        count += 1
        emm[i] = (sum(data[dolni_index: i + 1]*gamma[dolni_index: i + 1][::-1]))*(1/count)
    for i in range(window, len(data)):
        dolni_index = i + 1 - window
        emm[i] = (sum(data[dolni_index: i + 1]*gamm))*(1/count)

@guvectorize(['void(float64[:], int64, float64[:])'], '(n),()->(n)')
def moving_mean(a, window, out):
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
def moving_variance(data, okno, rozptyl):
    mm = moving_mean(data, okno)
    dolni_index = 0
    count = 0
    for i in range(okno):
        count += 1
        rozptyl[i] = sum((data[dolni_index: i + 1] - mm[i])**2)*(1/count)
    for i in range(okno, len(data)):
        dolni_index = i + 1 - okno
        rozptyl[i] = sum((data[dolni_index: i + 1] - mm[i])**2)*(1/count)

"""
def savitzky_golay_filter(data, okno, rad, deriv=0):
    # data, okno - delka useku, řád polynomu, řád derivace

    #předběžná kontrola možných problémů
    if okno % 2 != 1 or okno < 1:
        raise TypeError("okno musí být kladné liché číslo")
    if okno < rad + 2:
        raise TypeError("okno je příliž male pro polynom řádu %i" %rad)

    rozmezi_radu = range(rad+1)
    pulokno = (okno -1) // 2

    # předpočítat koeficienty
    b = np.mat([[k**i for i in rozmezi_radu] for k in range(-pulokno, pulokno+1)])
    m = np.linalg.pinv(b).A[deriv] * factorial(deriv)

    # zablokovat signál v extrémech s hodnotami převzatými ze samotného signálu
    firstvals = data[0] - np.abs(data[1:pulokno+1][::-1] - data[0])
    lastvals = data[-1] + np.abs(data[-pulokno-1:-1][::-1] - data[-1])
    data = np.concatenate((firstvals, data, lastvals))

    return np.convolve(m[::-1], data, mode='valid')
"""

def savitzky_golay_filter(x, window_length, polyorder, pos_back=1, deriv=0, axis=-1,
                          mode='nearest'):

    if pos_back > window_length:
        raise ValueError("pozice není uvnitř okna")

    #okraje mám defaulte pomocí nearest => nakopíruje krajní body
    if mode not in ["mirror", "nearest", "wrap"]:
        raise ValueError("mode must be 'mirror', 'nearest' or 'wrap'")

    x = np.asarray(x)
    # Nastavli jsem, aby se koeficienty počítaly v posledním bodě -> pos = window_lenght-1
    coeffs = savgol_coeffs(window_length, polyorder, pos=window_length - pos_back, deriv=deriv)
    # dále používám stejnou konvoluci jako je v originále
    y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=0.0)

    return y

def rozptyl_od_poc_fce(data, a_prumer_od_poc):
    odchylka = np.zeros(len(data))
    for i in range(len(data)):
        odchylka[i] = (1 / (i + 1)) * sum((data[0 : i + 1] - a_prumer_od_poc[0 : i + 1]) ** 2)
    return odchylka

def srovnej(res, data, pocet_stavu=3):
    # já vlastně přehazju data tak aby byla co největší schoda s res (skutečné výsledky)
    # když za data zadám stavy dostanu pak vektor stavů přerovnaný podle nám známého řešení(res)
    # to znamená že se bych je pak mohl přiradit správně do tříd, pokud bude známo řešení dopředu
    if pocet_stavu == 2:
        [temp0, temp1] = np.unique(data)

        reverse = np.copy(data)
        reverse[reverse == temp0] = -1
        reverse[reverse == temp1] = temp0
        reverse[reverse == -1] = temp1

        right_sorted = [sum(res == data), sum(res == reverse)]
        vector = np.vstack((data, reverse))
        return([max(right_sorted), vector[np.argmax(right_sorted), :]])
    else:
        if len(np.unique(data)) == 2:
            #srovnej(res, data, 2)
            [temp0, temp1] = np.unique(data)
            temp2 = copy(temp1)
        else:
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
            j += 1
        return([max(right_sorted), vector[np.argmax(right_sorted), :]])

def Accuracy(výsledek, stavy, pocet_stavu, srovnat=True):
    if srovnat:
        if len(výsledek) != len(stavy):
            print("stavy a výsledky nesouhlasí dimenze")
            return
            #print(Confusion_Matrix(výsledek, stavy, pocet_stavu))
        if pocet_stavu <= 2:
                #předpokládám že pří dvou stvech nebudu mít víc chyb než správných klasifikací
            součet = max(sum(výsledek == stavy), sum(výsledek != stavy))
            return [součet/len(stavy), len(stavy) - součet]
        else:
            přesnost = srovnej(výsledek, stavy)[0]
            return [přesnost / len(výsledek), int(len(výsledek) - přesnost)]
    else:
        sou = sum(výsledek == stavy)
        return [sou / len(stavy), int(len(stavy) - sou)]

def Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat = True):
    if srovnat == True:
        srovnaný = srovnej(výsledek, stavy, pocet_stavu)[1]
        #print(srovnaný)
        #print(výsledek)
    else:
        srovnaný = stavy
    tabulka = np.zeros((pocet_stavu, pocet_stavu), dtype='int64')
    for i in range(pocet_stavu):
        for j in range(pocet_stavu):
            tabulka[i, j] = sum(výsledek[k] == i and srovnaný[k] == j for k in range(len(výsledek)))
    return tabulka

def F_Measure(výsledek, stavy, pocet_stavu, srovnat=True):
    tabulka = Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat)
    FM = np.zeros(pocet_stavu)
    for k in range(pocet_stavu):
        FM[k] = 2 * tabulka[k, k] / (sum(tabulka[k, :]) + sum(tabulka[:, k]))
    # vracím [FM vektor, FM macro]
    # FM je F1_score('none') a suma je F1_score("macro") což je průměr F1 z všech tříd
    #print(tabulka)
    return [FM, sum(FM) / pocet_stavu]

def Precision_n_Recall(výsledek, stavy, pocet_stavu, srovnat = True):
    tabulka = Confusion_Matrix(výsledek, stavy, pocet_stavu, srovnat)
    precision = np.zeros(pocet_stavu)
    recall = np.zeros(pocet_stavu)
    for k in range(pocet_stavu):
        precision[k] = tabulka[k, k] / sum(tabulka[:, k])
        recall[k] = tabulka[k, k] / sum(tabulka[k, :])
        if isnan(precision[k]):
            precision[k] = 0
            #print("Precision was NaN ")
        if isnan(recall[k]):
            recall[k] = 0
            #print("Recall was NaN")
    #print(tabulka)
    return [precision, recall]

def Preprocessing(data, pocet_stavu, pocet_feature, labels):
    if np.shape(data)[0] < np.shape(data)[1]:
        raise TypeError("data nemají správný formát")

    #print("pocet rysu = ", pocet_feature, "tvar labels = ",
    #np.shape(labels), "tvar dat: ", np.shape(data))

    sorted_data_according_states = {}

    for state in range(pocet_stavu):
        sorted_data_according_states[state] = {}
        for feature in range(pocet_feature):
            sorted_data_according_states[state][feature] = []

    for label, _ in enumerate(labels): #label, _ in enumerate(labels)
        for feature in range(pocet_feature):
            sorted_data_according_states[labels[label]][feature].append(data[:, feature][label])

    means = np.zeros((pocet_stavu, pocet_feature))
    for i in sorted_data_according_states:
        for j in sorted_data_according_states[i]:
            means[i, j] = np.mean(sorted_data_according_states[i][j])

    variance = np.zeros((pocet_stavu, pocet_feature, pocet_feature))
    for i in sorted_data_according_states:
        for j in sorted_data_according_states[i]:
            variance[i, j, j] = np.var(sorted_data_according_states[i][j])

    return [means, variance]

def normalization(data, delka_useku=20, training_set=True):
    if training_set:
        return data/np.mean(data[:delka_useku])
    else:
        return data[delka_useku:]/np.mean(data[:delka_useku])

def Set_Noise(data, velikost_sumu = 1/40):
    noise = np.random.randn(len(data))
    return data + noise * velikost_sumu

def Set_Features(data_set,
                 delky_oken=[10, 10, 10, 10],
                 prvni_derivace=True,
                 druha_derivace=True,
                 suma_zleva=False,
                 aritmeticky_prumer=False,
                 rozptyl=False,
                 norm=False,
                 Training_set=True,
                 vypis_nastavene_vlastnosti=False):

    if vypis_nastavene_vlastnosti == True:
        print('Délka okna:', delky_oken,
              '\n Prvni_derivace:', prvni_derivace,
              '\n Druha_derivace:', druha_derivace,
              '\n Suma_zleva:', suma_zleva,
              '\n Aritmeticky_prumer:', aritmeticky_prumer,
              '\n Rozptyl:', rozptyl)

    if norm:
        X = normalization(np.array(data_set), delka_useku=20, training_set=Training_set)
        '''
        Díky bool(training set) můžu v budoucnu vynechávat z predikce úsek podle, kterého normuju
        a to z důvodu, že v praxi na začátku predikce nebudu tento úsek znát, proto budu muset
        nejdříve udělat střední hodnotu "normalizačního úseku" a pak s její pomocí normovát
        následující data
        '''
    else:
        X = np.array(data_set)

    if len(delky_oken) != 4:
        raise ValueError("delky oken musí být typu list se čtyřmi prvky")

    XX = np.copy(X)
    # samotné XX je jen data_set
    # do X jsou odteď přidané i feature

    if prvni_derivace == True:
        #Dx1 = Derivace(XX,1)
        Dx1 = savitzky_golay_filter(XX, 9, 3, 1)
        X = np.vstack([X, Dx1])

    if druha_derivace == True:
        #Dx1 = Derivace(XX,1)
        #Dx2 = Derivace(Dx1,1)
        Dx2 = savitzky_golay_filter(XX, 9, 3, 2)
        X = np.vstack([X, Dx2])

    for delky in delky_oken[:-1]:
        if delky != 0:
            if suma_zleva == True:
                Suma_L = exp_moving_mean(XX, delky)
                #[suma_zleva_fce(XX, x, delka_okna) for x in range(len(XX))]
                X = np.vstack([X, Suma_L])

            if aritmeticky_prumer == True:
                #Arit_Pr = [aritmeticky_prumer_fce(XX, x, delka_okna) for x in range(len(XX))]
                Arit_Pr = moving_mean(XX, delky)
                X = np.vstack([X, Arit_Pr])


    if rozptyl == True:
        Rozptyl = moving_variance(XX, delky_oken[3])
        X = np.vstack([X, Rozptyl])

    # transponuju teď už matici původního data setu a features
    return (X.T, XX)

def make_matrix(data, okna):
    if not isinstance(okna, list) or len(okna) < 4:
        raise ValueError("okna musí být list s nejméně 4 prvky")
    if not isinstance(data, list):
        data = [data]

    W = 3 + 2*len(okna[0]) + 2*len(okna[1]) + 2*len(okna[2]) + len(okna[3])
    out = np.zeros((W, 1))
    for d in data:
        norm_d = normalization(d[1], delka_useku=20, training_set=True)
        mat = np.asarray(norm_d)
        mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, deriv=1)))
        mat = np.vstack((mat, savitzky_golay_filter(norm_d, 9, 2, pos_back=5, deriv=2)))
        for mm0 in okna[0]:
            mat = np.vstack((mat, moving_mean(norm_d, mm0)))
        for mm1 in okna[1]:
            mat = np.vstack((mat, moving_mean(norm_d, mm1)))
        for mm2 in okna[2]:
            mat = np.vstack((mat, moving_mean(norm_d, mm2)))
        for emm0 in okna[0]:
            mat = np.vstack((mat, exp_moving_mean(norm_d, emm0)))
        for emm1 in okna[1]:
            mat = np.vstack((mat, exp_moving_mean(norm_d, emm1)))
        for emm2 in okna[2]:
            mat = np.vstack((mat, exp_moving_mean(norm_d, emm2)))
        for mv0 in okna[3]:
            mat = np.vstack((mat, moving_variance(norm_d, mv0)))
        #print("W = ", W, "mat = ", np.shape(mat))
        out = np.hstack((out, mat))
    return out[:, 1:].T

def train_and_predict(model, train, test, lengths, labels, unsupervised):
    warnings.filterwarnings('ignore')
    if unsupervised:
        model.fit(train)
    else:
        model.startprob_ = np.array([0, 1, 0])
        model.means_, model.covars_ = Preprocessing(train, 3,
                                                    np.shape(train)[1],
                                                    labels)
        model.fit(train, lengths)
        tm = copy(model.transmat_[1, 2])
        model.transmat_[1, 2] = 0
        model.transmat_[1, 1] = model.transmat_[1, 1] + tm
        del tm

    return model.predict(test)

def score(states, results, unsupervised):
    [Acc, M] = Accuracy(results, states, 3, unsupervised)
    [F, F_a] = F_Measure(results, states, 3, unsupervised)
    [P, R] = Precision_n_Recall(results, states, 3, unsupervised)
    return [Acc, M, F, F_a, P, R]

def vrat_delku_oken(delka_oken, cc, ran):
    w0 = len(delka_oken[0])
    w1 = len(delka_oken[1])
    w2 = len(delka_oken[2])

    if cc[0] < w0:
        O0 = delka_oken[0][cc[0]]
    else:
        O0 = 0

    if cc[1] < w1:
        O1 = delka_oken[1][cc[1]]
    else:
        O1 = 0

    if cc[2] < w2:
        O2 = delka_oken[2][cc[2]]
    else:
        O2 = 0

    return((O0, O1, O2, ran))

def validuj(model, train_data, test_data, delka_okna=[], parametry=[], unsupervised=True):
    """funkce validuj pro testování kombinací rysů"""
    if len(delka_okna) != 4:
        raise ValueError("delky oken musí být typu list se čtyřmi prvky")
    if len(parametry) != 5 and len(np.unique(parametry)) > 2 and len(parametry) != 0:
        raise ValueError("Parametry musí být list s pěti prvky typu bool (nebo 1,0)")

    warnings.filterwarnings('ignore')

    lengths = np.zeros(len(train_data), dtype=int)
    for i in range(len(train_data)):
        lengths[i] = len(train_data[i][1])

    #Nastavení labelů k datům
    if not unsupervised:
        labels = train_data[0][2]
        for lab in train_data[1:]:
            labels = np.hstack((labels, lab[2]))
        labels = labels.T

    Labely = test_data[0][2]
    if parametry:
        for lab in test_data[1:]:
            Labely = np.hstack((Labely, lab[0][2]))

        training_data = Set_Features(train_data[0][1], delka_okna,
                                     parametry[0], parametry[1],
                                     parametry[2], parametry[3], parametry[4],
                                     norm=True)[0]
        for train in train_data[1:]:
            training_data = np.vstack((training_data,
                                       Set_Features(train[1], delka_okna,
                                                    parametry[0], parametry[1],
                                                    parametry[2], parametry[3],
                                                    parametry[4],
                                                    norm=True)[0]))

        testing_data = Set_Features(test_data[0][1], delka_okna,
                                    parametry[0], parametry[1], parametry[2],
                                    parametry[3], parametry[4],
                                    norm=True)[0]
        for test in test_data[1:]:
            testing_data = np.vstack((testing_data[0][1],
                                      Set_Features(test, delka_okna,
                                                   parametry[0], parametry[1],
                                                   parametry[2], parametry[3],
                                                   parametry[4],
                                                   norm=True)[0]))

        CLF = copy(model)
        if unsupervised:
            stf = time.time()
            CLF.fit(training_data)
            endf = time.time()
        else:
            """
            Nejedná se tak úplně o supervised verzi.
            Spíše je to unsupervised s předpočítáním středních hodnot a covariančních matic
            """
            stf = time.time()
            CLF.startprob_ = np.array([0, 1, 0])
            CLF.means_, CLF.covars_ = Preprocessing(training_data, 3,
                                                    np.shape(training_data)[1],
                                                    labels)
            CLF.fit(training_data, lengths)
            endf = time.time()
            tm = copy(CLF.transmat_[1, 2])
            CLF.transmat_[1, 2] = 0
            CLF.transmat_[1, 1] = CLF.transmat_[1, 1] + tm
            del tm
            proba = CLF.predict_proba(testing_data)

        stp = time.time()
        states = CLF.predict(testing_data)
        endp = time.time()

        [acc, mis] = Accuracy(Labely, states, 3, unsupervised)
        [f, fa] = F_Measure(Labely, states, 3, unsupervised)
        [p, r] = Precision_n_Recall(Labely, states, 3, unsupervised)

        panda = list(zip([tuple(parametry), 0], [delka_okna, 0], [acc, 0], [mis, 0], [f[0], 0],
                         [f[1], 0], [f[2], 0], [fa, 0], [p[0], 0], [p[1], 0], [p[2], 0],
                         [r[0], 0], [r[1], 0], [r[2], 0]))

        dpanda = pd.DataFrame(data=panda, columns=['Kombinace rysů', 'délka úseku', 'Accuracy',
                                                   'Chyby', 'F míra stavu 0', 'F míra stavu 1',
                                                   'F míra stavu 2', 'F míra průměrná',
                                                   'Precision stavu 0', 'Precision stavu 1',
                                                   'Precision stavu 2', 'Recall stavu 0',
                                                   'Recall stavu 1', 'Recall stavu 2'])
        del training_data, testing_data, f, fa, acc, mis, p, r #, states
        return dpanda, states, endf-stf, endp-stp, CLF, proba
    else:
        #kombinacee všech možných rysů a délek oken
        [combinace, accuracy, chyby, F0, F1, F2, F_average, P0, P1, P2, R0, R1, R2, okno] = \
                                    [[], [], [], [], [], [], [], [], [], [], [], [], [], []]

        all_combos = ((2**5 - 1) * (len(delka_okna[0]) + 1) * (len(delka_okna[1]) + 1) \
                        * (len(delka_okna[2]) + 1) - 1) * len(delka_okna[3])

        print("počet všech možných kombinací je ", all_combos)

        bar = progressbar.ProgressBar(maxval=all_combos,
                                      widgets=[progressbar.Bar('#', '[', ']'),
                                               ' ', progressbar.Percentage()])
        bar.start()
        iterace = 0
        error = 0

        for MM1 in it.chain([0], delka_okna[0]):
            for MM2 in it.chain([0], delka_okna[1]):
                for MM3 in it.chain([0], delka_okna[2]):
                    if (MM1 + MM2 + MM3) == 0:
                        continue
                    for RM in delka_okna[3]:
                        for combin in it.product([0, 1], repeat=5):
                            try:
                                # repeat je počet možných feature, který lze použít
                                if combin == (0, 0, 0, 0, 0):
                                    continue
                                combinace.append(combin)
                                okno.append((MM1, MM2, MM3, RM))

                                training_data = Set_Features(train_data[0][1], [MM1, MM2, MM3, RM],
                                                             combin[0], combin[1], combin[2],
                                                             combin[3], combin[4],
                                                             norm=True)[0]
                                for train in train_data[1:]:
                                    training_data = np.vstack((training_data,
                                                               Set_Features(train[1],
                                                                            [MM1, MM2, MM3, RM],
                                                                            combin[0], combin[1],
                                                                            combin[2], combin[3],
                                                                            combin[4],
                                                                            norm=True)[0]))

                                testing_data = Set_Features(test_data[0][1], [MM1, MM2, MM3, RM],
                                                            combin[0], combin[1], combin[2],
                                                            combin[3], combin[4],
                                                            norm=True)[0]
                                for test in test_data[1:]:
                                    testing_data = np.vstack((testing_data,
                                                              Set_Features(test[1],
                                                                           [MM1, MM2, MM3, RM],
                                                                           combin[0], combin[1],
                                                                           combin[2], combin[3],
                                                                           combin[4],
                                                                           norm=True)[0]))

                                CLF = copy(model)
                                if unsupervised:
                                    CLF.fit(training_data)
                                else:
                                    """
                                    Nejedná se tak úplně o supervised verzi.
                                    Spíše je to unsupervised s předpočítáním středních hodnot
                                    a covariančních matic
                                    """
                                    CLF.startprob_ = np.array([0, 1, 0])
                                    CLF.means_, CLF.covars_ = Preprocessing(training_data, 3,
                                                                            np.shape(training_data)[1],
                                                                            labels)
                                    CLF.fit(training_data, lengths)
                                    tm = copy(CLF.transmat_[1, 2])
                                    CLF.transmat_[1, 2] = 0
                                    CLF.transmat_[1, 1] = CLF.transmat_[1, 1] + tm
                                    del tm

                                states = CLF.predict(testing_data)

                                [acc, mis] = Accuracy(Labely, states, 3, unsupervised)
                                accuracy.append(acc)
                                chyby.append(mis)

                                [f, fa] = F_Measure(Labely, states, 3, unsupervised)
                                F0.append(f[0])
                                F1.append(f[1])
                                F2.append(f[2])
                                F_average.append(fa)

                                [p, r] = Precision_n_Recall(Labely, states, 3, unsupervised)
                                P0.append(p[0])
                                P1.append(p[1])
                                P2.append(p[2])
                                R0.append(r[0])
                                R1.append(r[1])
                                R2.append(r[2])

                                del CLF, training_data, testing_data, f, fa, acc, mis, p, r, states
                                iterace += 1
                            #print(iterace)
                                bar.update(iterace)
                            except ValueError:
                                accuracy.append(0)
                                chyby.append(0)
                                F0.append(0)
                                F1.append(0)
                                F2.append(0)
                                F_average.append(0)
                                P0.append(0)
                                P1.append(0)
                                P2.append(0)
                                R0.append(0)
                                R1.append(0)
                                R2.append(0)

                                del CLF, training_data, testing_data
                                iterace += 1
                                error += 1
                                bar.update(iterace)
                                continue


        bar.finish()
        panda = list(zip(combinace, okno, accuracy, chyby, F0, F1, F2, F_average,
                         P0, P1, P2, R0, R1, R2))
        dpanda = pd.DataFrame(data=panda,
                              columns=['Kombinace rysů', 'délky úseku', 'Accuracy', 'Chyby',
                                       'F míra stavu 0', 'F míra stavu 1', 'F míra stavu 2',
                                       'F míra průměrná', 'Precision stavu 0', 'Precision stavu 1',
                                       'Precision stavu 2', 'Recall stavu 0', 'Recall stavu 1',
                                       'Recall stavu 2'])
        print(error)
        return dpanda


def validace_new(main_model, train_data, test_data, delka_okna, unsupervised=True):
    """
        při změně parametrů je potřeba přepsat přiřazování a maxval v progrssbaru
    """
    #4107
    bar = progressbar.ProgressBar(maxval=4263,
                                  widgets=[progressbar.Bar('#', '[', ']'),
                                           ' ', progressbar.Percentage()])
    bar.start()

    st = time.time()


    labels = train_data[0][2]
    for lab in train_data[1:]:
        labels = np.hstack((labels, lab[2]))
    labels = labels.T

    #print("tvar labelů", np.shape(labels))
    #print(labels)

    w0 = len(delka_okna[0])
    w1 = len(delka_okna[1])
    w2 = len(delka_okna[2])
    W = w0 + w1 + w2

    lengths = np.zeros(len(train_data), dtype=int)
    for i, j in enumerate(train_data):
        lengths[i] = len(train_data[i][1])

    train_matrix = make_matrix(train_data, delka_okna)
    test_matrix = make_matrix(test_data, delka_okna)
    #print(np.shape(train_matrix), np.shape(test_matrix))

    [comb, acc, miss, F0, F1, F2, F_avr, P0, P1, P2, R0, R1, R2, okno] = [[], [], [], [], [], [],
                                                                          [], [], [], [], [], [],
                                                                          [], []]
    iterace = 0
    #okna = [[1, 2], [2, 3], [3, 3], range(12)]
    #okna = [[1],[2],[3],range(12)]
    for combin in it.product([0, 1], repeat=5):
        if combin == (0, 0, 0, 0, 0):
            continue
        train = np.asarray(train_matrix[:, 0])
        test = np.asarray(test_matrix[:, 0])

        if combin[0]:
            train = np.vstack((train, train_matrix[:, 1]))
            test = np.vstack((test, test_matrix[:, 1]))
        if combin[1]:
            train = np.vstack((train, train_matrix[:, 2]))
            test = np.vstack((test, test_matrix[:, 2]))
        if combin[2] == 1 or combin[3] == 1:
            for cc in it.product([0, 1, 2], repeat=3):
                #if cc == (2, 2, 2):
                #    continue
                train1 = np.asarray(train)
                test1 = np.asarray(test)

                #print(cc)
                if combin[2] == 1:
                    if cc[0] < w0:
                        train1 = np.vstack((train1, train_matrix[:, 3 + cc[0]]))
                        test1 = np.vstack((test1, test_matrix[:, 3 + cc[0]]))
                    if cc[1] < w1:
                        train1 = np.vstack((train1, train_matrix[:, 3 + w0 + cc[1]]))
                        test1 = np.vstack((test1, test_matrix[:, 3 + + w0 + cc[1]]))
                    if cc[2] < w2:
                        train1 = np.vstack((train1, train_matrix[:, 3 + w0 + w1 + cc[2]]))
                        test1 = np.vstack((test1, test_matrix[:, 3 + w0 + w1 + cc[2]]))

                if combin[3] == 1:
                    if cc[0] < w0:
                        train1 = np.vstack((train1, train_matrix[:, 3 + W + cc[0]]))
                        test1 = np.vstack((test1, test_matrix[:, 3 + W + cc[0]]))
                    if cc[1] < w1:
                        train1 = np.vstack((train1, train_matrix[:, 3 + W + w0 + cc[1]]))
                        test1 = np.vstack((test1, test_matrix[:, 3 + W + w0 + cc[1]]))
                    if cc[2] < w2:
                        train1 = np.vstack((train1, train_matrix[:, 3 + W + w0 + w1 + cc[2]]))
                        test1 = np.vstack((test1, test_matrix[:, 3 + W + w0 + w1 + cc[2]]))
                if combin[4] == 1:
                    for i, j in enumerate(delka_okna[-1], 0):
                        train2 = np.asarray(train1)
                        test2 = np.asarray(test1)
                        train2 = np.vstack((train2, train_matrix[:, 3 + 2*W + i]))
                        test2 = np.vstack((test2, test_matrix[:, 3 + 2*W + i]))
                        model = copy(main_model)
                        #print("combin ", combin, "cc ", cc)
                        states = train_and_predict(model, train2.T, test2.T, lengths, labels,
                                                   unsupervised)

                        comb.append(combin)
                        okno.append(vrat_delku_oken(delka_okna, cc, j))
                        [a, m, f, f_a, p, r] = score(states, test_data[0][2], unsupervised)
                        acc.append(a)
                        miss.append(m)
                        F0.append(f[0])
                        F1.append(f[1])
                        F2.append(f[2])
                        F_avr.append(f_a)
                        P0.append(p[0])
                        P1.append(p[1])
                        P2.append(p[2])
                        R0.append(r[0])
                        R1.append(r[1])
                        R2.append(r[2])

                        del model, states, train2, test2, a, m, f, f_a, p, r
                        iterace += 1
                        bar.update(iterace)
                else:
                    model = copy(main_model)
                    #print("combin ", combin, "cc ", cc)
                    states = train_and_predict(model, train1.T, test1.T, lengths, labels,
                                               unsupervised)
                    comb.append(combin)
                    okno.append(vrat_delku_oken(delka_okna, cc, 0))
                    [a, m, f, f_a, p, r] = score(states, test_data[0][2], unsupervised)
                    acc.append(a)
                    miss.append(m)
                    F0.append(f[0])
                    F1.append(f[1])
                    F2.append(f[2])
                    F_avr.append(f_a)
                    P0.append(p[0])
                    P1.append(p[1])
                    P2.append(p[2])
                    R0.append(r[0])
                    R1.append(r[1])
                    R2.append(r[2])

                    del model, states, train1, test1, a, m, f, f_a, p, r
                    iterace += 1
                    bar.update(iterace)

        elif combin[4] == 1 and sum(combin[2:-1]) == 0:
            for i, j in enumerate(delka_okna[-1], 0):
                train1 = np.asarray(train)
                test1 = np.asarray(test)
                train1 = np.vstack((train1, train_matrix[:, 3 + 2*W + i]))
                test1 = np.vstack((test1, test_matrix[:, 3 + 2*W + i]))
                #print("tvar dat před train_and_predict je ", np.shape(train1))
                model = copy(main_model)
                #print("combin ", combin)
                states = train_and_predict(model, train1.T, test1.T, lengths, labels, unsupervised)
                comb.append(combin)
                okno.append((0, 0, 0, j))
                [a, m, f, f_a, p, r] = score(states, test_data[0][2], unsupervised)
                acc.append(a)
                miss.append(m)
                F0.append(f[0])
                F1.append(f[1])
                F2.append(f[2])
                F_avr.append(f_a)
                P0.append(p[0])
                P1.append(p[1])
                P2.append(p[2])
                R0.append(r[0])
                R1.append(r[1])
                R2.append(r[2])

                del model, states, train1, test1, a, m, f, f_a, p, r
                iterace += 1
                bar.update(iterace)
        else:
            model = copy(main_model)
            #print("combin ", combin)
            states = train_and_predict(model, train.T, test.T, lengths, labels, unsupervised)
            comb.append(combin)
            okno.append((0, 0, 0, 0))
            [a, m, f, f_a, p, r] = score(states, test_data[0][2], unsupervised)
            acc.append(a)
            miss.append(m)
            F0.append(f[0])
            F1.append(f[1])
            F2.append(f[2])
            F_avr.append(f_a)
            P0.append(p[0])
            P1.append(p[1])
            P2.append(p[2])
            R0.append(r[0])
            R1.append(r[1])
            R2.append(r[2])

            del model, states, train, test, a, m, f, f_a, p, r
            iterace += 1
            bar.update(iterace)

    bar.finish()
    panda = list(zip(comb, okno, acc, miss, F0, F1, F2, F_avr,
                     P0, P1, P2, R0, R1, R2))
    dpanda = pd.DataFrame(data=panda,
                          columns=['Kombinace rysů', 'délky úseku', 'Accuracy', 'Chyby',
                                   'F míra stavu 0', 'F míra stavu 1', 'F míra stavu 2',
                                   'F míra průměrná', 'Precision stavu 0', 'Precision stavu 1',
                                   'Precision stavu 2', 'Recall stavu 0', 'Recall stavu 1',
                                   'Recall stavu 2'])
    en = time.time()
    print(en-st)
    return dpanda


#isinstance(<var>, int)
