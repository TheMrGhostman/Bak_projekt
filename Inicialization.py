import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import Classification as CL
import warnings

# načtu data k tréninku
# freature můž předpřipravit dopředu nebo je načíst dále
data = CL.Set_Features(data, True, 1/40, 5, 0, 0, 1, 1, 0, 0, 0)

training_data = np.vstack((data1,data2,....))

HMM_klasifikace = GaussianHMM(pocet_stavu)

#buď Unsupervised
#HMM_klasifikace.fit(training_data)

#nebo Supervised
#HMM_klasifikace.fit(training_data, real_lables)
#######################################


#můžu zkusit klasifikovat testovací data
states = HMM_klasifikace.predict(testing_data)
print(states)
