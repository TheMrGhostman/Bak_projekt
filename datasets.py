import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import chain
from bunch import Bunch
from copy import copy


path = os.getcwd() + "/Data_npy/Chosen ones/"

def shape(data):
    """
    náhrada np.shape schopná zmapovat a spočítat všechny rozsahy (shape)
    jednotlivých vrsetv struktury v níž načítám a používám datasety (resp. signály)
    - hlavně je schopna poradit si s "lists in list"
    """
    signal_layers = []
    datapoints = []
    for signal in data:
        signal_layers.append(len(signal))
        if len(np.shape(signal)) != 1:
            datapoints.append(len(signal[0]))

    if len(np.unique(signal_layers)) == 1:
        return((len(data), len(data[0]), datapoints, sum(datapoints)))
    else:
        return((len(data), signal_layers, datapoints, sum(datapoints)))

def get_layer(data, layer):
    output = []
    for signal in data:
        output.append(signal[layer])
    return output

def vizualize(bunch):
    plt.figure("Vyobrazení celého datasetu")
    posun = 0
    for i in bunch.H_alpha:
        plt.plot(np.arange(len(i))+ posun, i, '-', lw=0.5)
        posun = posun + len(i)
    plt.xlabel("x - points count (not time)")
    plt.ylabel(r"$H_{\alpha}$")
    plt.show()

def load_dataset(name, dataset_only=False):
    """
    Snaha o zjednodušení načítání dat pro výpočty.
    """

    signals = np.load(os.getcwd() + "/Data_npy/{}.npy".format(name)).tolist()

    dataset = [np.load(path + data) for data in chain(signals)]

    time = get_layer(dataset, 0)
    H_alpha = get_layer(dataset, 1)
    labels = get_layer(dataset, 2)

    return Bunch(data=dataset, time=time, H_alpha=H_alpha, labels=labels,
                 signals=signals, shape=shape(dataset))



def PrepareCrossFold(data):
    """
    Funkce připravuje data pro použití při křížové validaci po signálech.
    Použitý poměr je 9:1 (trénink:testování). Vrací data ve formě, kterou
    vyžaduje funkce Classification.validace_hromadna().

    Output: Vrací dva seznamy -> train ... trénovací data
                                 test  ... testovací data
    """
    train = np.empty([10]).tolist()
    test = np.empty([10]).tolist()

    for i, j in enumerate(data):
        if i == 0:
            train[i] = copy(data[:-1])
            test[i] = [copy(data[-1])]
        else:
            train[i] = copy(data[:-(i+1)]) + copy(data[-i:])
            test[i] = [copy(data[-(i+1)])]
    return train, test




def unit_test():
    dat = load_dataset(name="first_dataset")
    #print(dat.data)
    print(shape(dat.H_alpha))
    print(shape(dat.labels))
    print(dat.shape)
    vizualize(dat)
    #print(shape(dat.train()))


#unit_test()
