#import Classification as cl
import numpy as np
import matplotlib.pyplot as plt
import os


##pwd = os.getcwd()
##pwd
##for i in reversed(pwd):
##    if i == '/':
##        break
##    else:
##        pwd = pwd[:len(pwd)-1]
##pwd
##pwd += "Bakalářka"
##pwd

import Classification as cl

def aritm(data):
    ar = []
    for i in range(len(data)):
        ar.append(np.mean(data[:i+1]))
    return(np.array(ar))



X = np.load("Synteticka_data_sum_0.025.npy")

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
ax = plt.gca()

ax.plot(X, label = "syntetická data")
ax.plot([cl.aritmeticky_prumer_fce(X, x, 5) for x in range(len(X))]-np.mean(X), label = "úsekový aritmetický průměr")
ax.plot(aritm(X)-2*np.mean(X), color = "lime", label = "aritmetický průměr")
ax.plot(cl.rozptyl_od_poc_fce(X, aritm(X)))
ax.legend()
#ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

usek  =[cl.aritmeticky_prumer_fce(X, x, 5) for x in range(len(X))]
ar = aritm(X)

gmax = max([max(X), max(usek), max(ar)])+((max(X)-min(X))/10)
gmin = min([min(X), min(usek), min(ar)])-((max(X)-min(X))/10)

fig, axs = plt.subplots(3,1, sharex = True)
fig.subplots_adjust(hspace = 0)
axs[0].plot(X, label = "data")
axs[0].set_yticks(np.around(np.arange(gmin, gmax,(gmax-gmin)/5), decimals = 2))
axs[0].set_ylim(gmin,gmax)
axs[0].legend(loc = "lower right")
axs[1].plot(usek, label = "úsekový aritmetický průměr", color = "orange")
axs[1].set_yticks(np.around(np.arange(gmin, gmax,(gmax-gmin)/5), decimals = 2))
axs[1].set_ylim(gmin-0.05,gmax)
axs[1].legend(loc = "lower right")
axs[2].plot(ar, label = 'standartní aritmetický průměr', color  = "red")
axs[2].set_yticks(np.around(np.arange(gmin, gmax,(gmax-gmin)/5), decimals = 2))
axs[2].set_ylim(gmin,gmax-0.02)
axs[2].legend(loc = "lower right")
plt.show()



for i in [1/2,1/3,1/4,1/5,1/6]:
    plt.plot(X**i)
plt.show()


def odchylka(data):
    ex = aritm(data)
    o = np.cumsum(ex-data)
    return(o)

plt.plot(odchylka(X))
plt.show()

x = np.array([1, 2, 4, 7, 11, 16])
y = np.gradient(x)
y
y = np.gradient(x,3)
y

DX = np.gradient(X)
plt.plot(X)
plt.plot(DX)
plt.show()

XXX = X+np.random.randn(len(X))*1/4
plt.plot(XXX)
plt.plot(np.gradient(XXX))
plt.plot([cl.aritmeticky_prumer_fce(X, x, 5) for x in range(len(X))]-np.mean(X))
plt.show()
