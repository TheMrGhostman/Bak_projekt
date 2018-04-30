import numpy as np
import matplotlib.pyplot as plt
import os
#/Bak_projekt-IPP-branch

way = os.getcwd() + "/Labeled Sets/"
way
dirr = os.listdir(way)

way1 = way + "Chosen ones/"
#print(dirr)

np.load(way + dirr[0])

def zobraz_all(wayy, directory):
    for file in directory:
        if not "info" in file:
            plt.figure("data: " + str(file))

            X = np.load(wayy + file)
            plt.plot(X[0], X[1])
            plt.scatter(X[0], X[1] , c = X[2], cmap = plt.cm.plasma)
            plt.show()

#zobraz_all(way,dirr)

def Choose(wayy, root_directory, wayy1):
    remake = []
    for file in root_directory:
        if not "info" in file:
            tf = input("chces ulozit " + file + "?: ")
            if tf == "y":
                X = np.load(wayy + file)
                np.save(wayy1 + file, X)
            if tf == "n":
                re = input("chces upravit " + file + "?: ")
                if re == "y":
                    remake.append(file)
    return remake


def T_M(data, t0 = 1050, t1 = 1200, name = "nic",kresli = True):
    #data = cdb.get_signal("H_alpha: %s" %str(number))
    #hmode = cdb.get_signal("t_H_mode_start: %s" %str(number))
    #hend = cdb.get_signal("t_H_mode_end: %s" %str(number))

    time_window_mask = (data[0] > t0) & (data[0] < t1)
    t = data[0][time_window_mask]
    x0 = data[1][time_window_mask]
    la = data[2][time_window_mask]

    if kresli:
        plt.figure(name)
        plt.plot(t, x0)
        plt.scatter(t, x0 , c = la, cmap = plt.cm.plasma)
        plt.show()
    return np.vstack((t,x0,la))

def upravy(seznam,way,way1):
    opravy = []
    for file in seznam:
        tf = True
        XX = np.load(way + file)
        t0 = XX[0][0]
        t1 = XX[0][len(XX[0])-1]
        while tf:
            XX = T_M(XX, t0, t1,file)
            tf = input("hotovo? ")
            t0 = input("t0 ")
            t1 = input("t1 ")
            if t0 == 0:
                t0 = XX[0][0]
            if t1 == 0:
                t1 = XX[0][len(XX[0])-1]
        if input("ulozit ? "):
            np.save(way1 + file, XX)
        else:
            if input("dalsi opravy? "):
                opravy.append(file)
                np.save(way1 + "nutna_dalsi_oprava_" + file, XX)
    return opravy

#tmp = Choose(way,dirr,way1)
#np.save("upravit.npy", tmp)
dirr2 = os.listdir(way1)
print(len(dirr2)/2)

X = np.load("upravit.npy")
print(len(X))
#if input("zobraz"):
#    print(X)

tmp = upravy(X, way,way1)
np.save("opravy.npy",tmp)
