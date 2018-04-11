import numpy as np
import Classification as CL
import os
#import python_pwd
way = os.getcwd() + "/Data_npy/"


data = np.hstack((3*np.ones(10), np.zeros(5), 2 * np.ones(10)))

def Test_Derivace():
    test_data = [1, 2, 4, 7, 11, 16]
    derivace = CL.Derivace(test_data)
    if(np.allclose(derivace, np.gradient(test_data))):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_SZL():
    test_data = np.array([CL.suma_zleva_fce(np.ones(20), i, 10) for i in range(len(np.ones(20)))])

    real_result = np.load(way + "Unit_test_pro_SZLF.npy")

    if(np.allclose(test_data,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_ARPF():
    test_data = np.array([CL.aritmeticky_prumer_fce(data, i, 5)
                          for i in range(25)])

    real_result = np.load(way + "Unit_test_pro_ARPF.npy")

    if(np.allclose(test_data,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_RF():
    test_data = CL.rozptyl_fce(data, 5)

    real_result = np.load(way + "Unit_test_pro_RF.npy")

    if(np.allclose(test_data,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_ROPF():
    aritmetiky_prumer_od_pocatku = np.cumsum(data)/(np.arange(len(data))+1)

    test_data = CL.rozptyl_od_poc_fce(data, aritmetiky_prumer_od_pocatku)

    real_result = np.load(way + "Unit_test_pro_ROPF.npy")

    if(np.allclose(test_data,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_srovnej():
    res_data = np.load(way + 'Unit_test_pro_srovnej_result.npy')
    data = np.load(way + 'Unit_test_pro_srovnej_stavy.npy')
    real_result = np.load(way + 'Unit_test_pro_srovnej_kontrola.npy')
    srovnani = CL.srovnej(data, res_data)
    #původně bylo CL.srovnej(res_data,data) ale po opravě chyby ve funkci srovnej se to musí zadávat takto
    if(srovnani[0] == 197 and np.allclose(srovnani[1], real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_Conf_Mat():
    res = np.array([0, 1, 2, 0, 1, 2])
    data = np.array([0, 1, 1, 0, 0, 1])
    real_result = np.array([2, 0, 0, 1, 1, 0, 0, 2, 0]).reshape(3,3)
    if np.allclose(CL.Confusion_Matrix(res, data, 3 , False),real_result):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_FM():
    res = np.hstack((np.ones(50),np.zeros(40)))
    data = np.hstack((np.zeros(40),np.ones(50)))
    #print(CL.Confusion_Matrix(res, data, 2))
    #print(CL.srovnej(res,data,2))
    #print("celý",CL.F_Measure(res, data, 2))
    FM = CL.F_Measure(res, data, 2)
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
    PaR = CL.Precision_n_Recall(res,data,3,False)
    real_result = np.load(way + "PaR_real_res.npy")
    if(np.allclose(PaR,real_result)):
        return("Funguje")
    else:
        return("Nefunguje")

def Test_klasifikuj():
    return(0)

def main_all():
    print("Proběhl test všech funkcí z modulu Classification." )
    print("Derivace: ", Test_Derivace(),
    "\nVáhová suma zleva: ", Test_SZL(),
    "\nÚsekový aritmetický průměr: ", Test_ARPF(),
    "\nÚsekový rozptyl: ", Test_RF(),
    "\nRozptyl od počátku: ", Test_ROPF(),
    "\nSorovnánvací funkce: ", Test_srovnej(),
    "\nConfusion matrix: ", Test_Conf_Mat(),
    "\nF-míra pro dva stavy: ", Test_FM(),
    "\nPrecision a Recall: ", Test_PaR())
