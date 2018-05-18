import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import Classification as CL
import K_Means as km
import warnings
import os

way = os.getcwd() + "/Data_npy/"

#
# # načtu data k tréninku
# # freature můž předpřipravit dopředu nebo je načíst dále
# data = CL.Set_Features(data, True, 1/40, 5, 0, 0, 1, 1, 0, 0, 0)
#
# training_data = np.vstack((data1,data2,....))
#
# HMM_klasifikace = GaussianHMM(pocet_stavu)
#
# #buď Unsupervised
# #HMM_klasifikace.fit(training_data)
#
# #nebo Supervised
# #HMM_klasifikace.fit(training_data, real_lables)
# #######################################
#
#
# #můžu zkusit klasifikovat testovací data
# states = HMM_klasifikace.predict(testing_data)
# print(states)

X = np.load(way + "Synteticka_data_sum_0.025.npy")
Y = CL.Moving_Mean(X,5)
#[CL.aritmeticky_prumer_fce(X, x, 5) for x in range(len(X))]
Z = CL.Exp_Moving_Mean(X,5)
#[CL.suma_zleva_fce(X, x, 5) for x in range(len(X))]
XX = np.vstack((Y,Z,X)).T
XX
# np.shape(XX)
# plt.scatter(Y,X)
# plt.show()

np.shape(Z)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(Y,Z,X)
plt.show()
km.Nakresly(XX,2, "3d")

clf = km.K_means(2)
clf.fit(XX)
predictions = []
for pred in XX:
    predictions.append(clf.Predict(pred))
print(predictions)
plt.plot(np.arange(len(X)), X, linewidth = 0.5)
plt.scatter(np.arange(len(X)), X, c = predictions, cmap = plt.cm.plasma)
plt.show()


ZZ = np.load(way + "No2.npy")
ZZ[0]
plt.plot(ZZ[0],ZZ[1])
plt.show()
CL.klasifikuj(ZZ[1], ZZ[0] , 3, False, 1 , 7 , 0,0,1,1,0)
lab = np.load(way + "No2_res.npy")
lab

No1 = np.load(way + "No3.npy")
plt.plot(No1[0],No1[1])
plt.show()


number = 1111111
hmode = np.arange(4)
print(["record number: " + str(number),"hmode", hmode])
mix = np.asarray(["Record number: ", str(number), "t_hmode: "])
np.hstack((np.asarray(["Record number: ", str(number), "t_hmode: "]),hmode))

d = np.linspace(-5,4,10)/10
d

x = np.random.randint(np.min(d)*10,np.max(d)*10, size=(10,3))
x

r = np.random.choice(len(x)-1,3)
r
x[r[0]]

Y = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1, 0.6], [9,11]])
initial_centroids = np.random.choice(np.shape(Y)[0]-1, 2)
centroids = np.zeros(2).tolist()
for i in range(2):
    centroids[i] = Y[initial_centroids[i]]
centroids
initial_centroids


Y = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1, 0.6], [9,11]])
Y
Y[0]
Y[initial_centroids[0]]
np.shape(Y)[0]-1
