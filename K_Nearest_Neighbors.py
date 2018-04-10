import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
from collections import Counter

data_set = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,6]]}
test = [5,7]

[[plt.scatter(ii[0],ii[1], s=100, color = i) for ii in data_set[i]] for i in data_set]
plt.scatter(test[0], test[1])
plt.scatter(4, 4)
plt.show()

data_set['k']
data_set['k'][0]
data_set['k'][0][0]

for i in data_set:
    print(data_set[i][1])

for i in data_set:
    for fet in data_set[i]:
        print(np.array(fet))

euclidean_distance = []
for i in data_set:
    for ii in range(len(data_set[i])):
        euclidean_distance.append(np.linalg.norm(np.array(data_set[i][ii])-np.array(test)))
        #print(data_set[i][ii])
euclidean_distance
del euclidean_distance

def K_Nearest_Neighbors(Data, toPredict, DistanceType = 'euclidean', k_neighbors = 3):
    if len(Data) >= k_neighbors:
        warnings.warn('Problém')
    Distance = []
    if DistanceType == 'euclidean':
        for shluk in Data:
            for bod in Data[shluk]:
                Distance.append([np.linalg.norm(np.array(bod)-np.array(toPredict)), shluk])
    elif DistanceType == 'manhattan':
        for shluk in Data:
            for bod in Data[shluk]:
                #manhattan = sum(abs(np.array(bod)-np.array(toPredict)))
                Distance.append([sum(abs(np.array(bod)-np.array(toPredict))), shluk])
    elif DistanceType == 'canberra':
        for shluk in Data:
            for bod in Data[shluk]:
                bod = np.array(bod)
                toPredict = np.array(toPredict)
                Distance.append([sum(abs(bod-toPredict)/(abs(bod)+abs(toPredict))), shluk])
    else:
        print('zatím není jiná možnost')
        return
    #votes = [i[1] for i in sorted(Distance)[:k_neighbors]]
    #Decision = Counter(votes).most_common(1)[0][0]
    return Counter([i[1] for i in sorted(Distance)[:k_neighbors]]).most_common(1)[0][0]
    #return Decision

print(K_Nearest_Neighbors(data_set, test, 'manhattan'))
print(K_Nearest_Neighbors(data_set, [4,4], 'manhattan'))
print(K_Nearest_Neighbors(data_set, [4,4], 'euclidean'))
print(K_Nearest_Neighbors(data_set, test, 'canberra'))
print(K_Nearest_Neighbors(data_set, [4,4], 'canberra'))

l = np.arange(10)
ll = np.ones(10)
ll-l
sum(abs(ll-l))
sum(abs(ll-l)/(abs(ll)+abs(l)))
np.linalg.norm(ll-l)
