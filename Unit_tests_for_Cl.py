import numpy as np
import Classification as Cl
import os
import math
#import python_pwd
way = os.getcwd() + "/Data_npy/"

global data
data = np.hstack((3*np.ones(10), np.zeros(5), 2 * np.ones(10)))

def Test_Derivace():
    test_data = [1, 2, 4, 7, 11, 16]
    derivace = Cl.Derivace(test_data,1)
    if(np.allclose(derivace, np.gradient(test_data))):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_SZL():
    test_data = Cl.Exp_Moving_Mean(np.ones(20),10)
    #np.array([Cl.suma_zleva_fce(np.ones(20), i, 10) for i in range(len(np.ones(20)))])

    real_result = np.load(way + "Unit_test_pro_SZLF.npy")

    if(np.allclose(test_data,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_ARPF():
    test_data = Cl.Moving_Mean(data, 5)
    real_result = np.load(way + "Unit_test_pro_ARPF.npy")

    if(np.allclose(test_data,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_RF():
    test_data = Cl.Moving_Variance(data, 5)
    real_result = np.load(way + "Unit_test_pro_RF.npy")

    if(np.allclose(test_data,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")


def Test_srovnej():
    res_data = np.load(way + 'Unit_test_pro_srovnej_result.npy')
    data = np.load(way + 'Unit_test_pro_srovnej_stavy.npy')
    real_result = np.load(way + 'Unit_test_pro_srovnej_kontrola.npy')
    srovnani = Cl.Srovnej(data, res_data)
    #původně bylo CL.srovnej(res_data,data) ale po opravě chyby ve funkci srovnej se to musí zadávat takto
    if(srovnani[0] == 197 and np.allclose(srovnani[1], real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_Conf_Mat():
    res = np.array([0, 1, 2, 0, 1, 2])
    data = np.array([0, 1, 1, 0, 0, 1])
    real_result = np.array([2, 0, 0, 1, 1, 0, 0, 2, 0]).reshape(3,3)
    if np.allclose(Cl.Confusion_Matrix(res, data, 3 , False),real_result):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_FM():
    res = np.hstack((np.ones(50),np.zeros(40)))
    data = np.hstack((np.zeros(40),np.ones(50)))
    #print(CL.Confusion_Matrix(res, data, 2))
    #print(CL.srovnej(res,data,2))
    #print("celý",CL.F_Measure(res, data, 2))
    FM = Cl.F_Measure(res, data, 2)
    #print("0" ,round(FM[0][0],3))
    #print("1" ,round(FM[0][1],3))
    #print("round",round(FM[1], 5))
    real_result = [0.889, 0.889, 0.88889]
    #print(real_result)
    if ((round(FM[0][0],3) == real_result[0]) and
    (round(FM[0][0],3) == real_result[1]) and
    (round(FM[1],5) == real_result[2])):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_PaR():
    #test Precision_n_Recall
    res = np.array([0, 1, 2, 0, 1, 2, 2])
    data = np.array([0, 1, 1, 0, 0, 1, 2])
    PaR = Cl.Precision_n_Recall(res,data,3,False)
    real_result = np.load(way + "PaR_real_res.npy")
    if(np.allclose(PaR,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_klasifikuj():
    return(0)

def Test_Acc():
    [acc, mis] = Cl.Accuracy(np.hstack((np.ones(10), 2 * np.ones(15), np.zeros(5))), np.hstack((np.ones(10),np.zeros(15), 2 * np.ones(5))), 3, True)
    [acc1, mis1] = Cl.Accuracy(np.hstack((np.ones(10), 2 * np.ones(15), np.zeros(5))),np.hstack((np.ones(10),np.zeros(15), 2 * np.ones(5))), 3, False)
    #print(acc, mis, acc1, mis1)
    if acc == 1.0 and mis == 0 and acc1 == 1/3 and mis1 == 20:
        return("Funguje")
    else:
        return("Nefunguje")

def main_all():
    print("Proběhl test všech funkcí z modulu Classification." )
    print("Derivace: ", Test_Derivace(),
    "\nVáhová suma zleva: ", Test_SZL(),
    "\nÚsekový aritmetický průměr: ", Test_ARPF(),
    "\nÚsekový rozptyl: ", Test_RF(),
    "\nSorovnánvací funkce: ", Test_srovnej(),
    "\nConfusion matrix: ", Test_Conf_Mat(),
    "\nF-míra pro dva stavy: ", Test_FM(),
    "\nPrecision a Recall: ", Test_PaR(),
    "\nAccuracy: ", Test_Acc())

main_all()
