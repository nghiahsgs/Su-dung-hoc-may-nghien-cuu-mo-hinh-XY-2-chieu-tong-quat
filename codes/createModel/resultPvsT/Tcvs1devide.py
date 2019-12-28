import numpy as np
import matplotlib.pyplot as plt
import sys

args=sys.argv
type=args[1]#xy
delta=args[2]#0.34


data_Tc=np.load('data_Tc.npy')

#Ve Tc theo 1/L
list_delta=['0.34','1.0']
list_type=['xy','angle','vortex','angle_vortex']
list_L=[12,16,24,32]
list_L_inverse=[1/L for L in list_L]

position_delta=list_delta.index(delta)
postiion_type=list_type.index(type)

list_Tc=data_Tc[:, postiion_type,position_delta]
#print(list_Tc)

plt.plot(list_L_inverse,list_Tc,'x')

plt.show()
