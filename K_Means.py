import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')

colors = 10*["g","r","c","b","k"]

Y = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1, 0.6], [9,11]])
plt.scatter(Y[:,0],Y[:,1], s = 100 , color = 'red')
plt.show()

class K_means:
    def __init__(self, k = 2, tol = 0.001, max_iterations = 300):
        self.k = k
        self.tol = tol
        self.max_iterations = max_iterations

    def fit(self, Data):
        self.centroids = {}
        np.random.shuffle(Data)
        #print(self.tol)
        for i in range(self.k):
            self.centroids[i] = Data[i]

        for i in range(self.max_iterations):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for features in Data:
                Distance = [np.linalg.norm(features - self.centroids[centr]) for centr in self.centroids]
                classification = Distance.index(min(Distance))
                self.classifications[classification].append(features)

            previous_centroids = dict(self.centroids)

            for c in self.classifications:
                #print("average" , np.average(self.classifications[c], axis = 0))
                #self.centroids[c] = np.average(self.classifications[c], axis = 0)
                self.centroids[c] = np.mean(self.classifications[c], axis = 0)

            convergence = True

            for d in self.centroids:
                #print(previous_centroids[d])
                #print(self.centroids[d])
                #print(np.sum((self.centroids[d]-previous_centroids[d])/previous_centroids[d]*100))
                if np.sum((self.centroids[d]-previous_centroids[d])/previous_centroids[d]*100)>0.001:
                    convergence = False

            if convergence == True:
                break

    def Predict(self, Data):
        Distance = [np.linalg.norm(Data-self.centroids[centr]) for centr in self.centroids]
        Prediction = Distance.index(min(Distance))
        return Prediction


# def data_prepare(data):
#     if np.s
#     if np.shape(data)[1]>np.shape(data)[0]:
#
#




def Nakresly(Data, skupiny, dims = '2d'):
    #Data = data_prepare(Data)
    barvy = 10 * ["g","r","c","b","k"]
    clf = K_means(skupiny)
    clf.fit(Data)
    if dims == '2d':
        for c in clf.centroids:
            plt.scatter(clf.centroids[c][0], clf.centroids[c][1], marker = "o",
            color = "blue", s = 100, linewidths = 5)
        ind = 0
        for cc in clf.classifications:
            color = barvy[ind]
            for x in clf.classifications[cc]:
                plt.scatter(x[0], x[1], marker = "x", color = color, s = 100, linewidths = 5)
            ind +=1
        plt.show()
        del ind
    elif dims == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        for c in clf.centroids:
            ax.scatter(clf.centroids[c][0], clf.centroids[c][1],clf.centroids[c][2], marker = "." ,
            color = "blue" , s = 100, linewidths = 5)
        ind = 0
        for cc in clf.classifications:
            color = barvy[ind]
            for x in clf.classifications[cc]:
                ax.scatter(x[0], x[1], x[2], marker = "x" , color = color , s = 100, linewidths = 5)
            ind +=1
        plt.show()
        del ind


# clf = K_means()
# clf.fit(Y)
#
# for c
#  in clf.centroids:
#    plt.scatter(clf.centroids[c][0], clf.centroids[c][1], marker = "o" ,
#    color = "blue" , s = 100, linewidths = 5)
#
# for c in clf.classifications:
#    color = colors[c]
#    for features in clf.classifications[c]:
#        plt.scatter(features[0], features[1], marker = "x" , color=color , s = 100, linewidths = 5)
#
# unknowns = np.array([[1,3],
#                    [8,9],
#                    [0,3],
#                    [5,4],
#                    [6,4]])
# for u in unknowns:
#    classify = clf.Predict(u)
#    plt.scatter(u[0], u[1], marker = "." , color = colors[classify], s = 100, linewidths = 5)
#
# plt.show()
#
#
YY = np.array([[1,2,3], [1.5,1.8,1.4], [5,8,7], [8,8,8], [1, 0.6, 0.8], [9,11,10]])
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(YY[:, 0], YY[:, 1], YY[:,2], s = 100 , color = 'red')
plt.show()

Nakresly(YY,2, '2d')
#clf = K_means()
#clf.fit(YY)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')

#for c in clf.centroids:
#    ax.scatter(clf.centroids[c][0], clf.centroids[c][1],clf.centroids[c][2], marker = "." ,
#    color = "blue" , s = 100, linewidths = 5)

#for c in clf.classifications:
#    color = colors[c]
#    for features in clf.classifications[c]:
#        ax.scatter(features[0], features[1], features[2], marker = "x" , color=color , s = 100, linewidths = 5)

#unknowns = np.array([[1,3,1],
#                    [8,9,7],
#                    [0,3,2],
#                    [5,4,5],
#                    [6,4,7]])
#for u in unknowns:
#    classify = clf.Predict(u)
#    ax.scatter(u[0], u[1], u[2], marker = "." , color = colors[classify], s = 100, linewidths = 5)

#plt.show()


YYY = np.random.randint(100, size = (40,3))
#Nakresly(YYY,'3d')
#clf = K_means()
#clf.fit(YYY)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(YYY[:,0], YYY[:,1], YYY[:,2], s = 100 , color = 'orange')
plt.show()

Nakresly(YYY,2,'3d')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')

#for c in clf.centroids:
#    ax.scatter(clf.centroids[c][0], clf.centroids[c][1],clf.centroids[c][2], marker = "." ,
#    color = "blue" , s = 100, linewidths = 5)

#for c in clf.classifications:
#    color = colors[c]
#    for features in clf.classifications[c]:
#        ax.scatter(features[0], features[1], features[2], marker = "x" , color=color , s = 100, linewidths = 5)

#plt.show()
