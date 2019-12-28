import sys
import numpy as np
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
def hamSoSanh2MangFlatten(A,B):
    #Neu kernel A lon hon kernel B thi return True
    rows=3
    cols=3
    for i_row in range(rows):
        for i_col in range(cols):
            element_A=A[i_row,i_col]
            element_B=B[i_row,i_col]
            if element_A>element_B:
                return True
            else:
                return False

    return False
def bubble_sort_list(L):
    n=len(L)
    for i in range(n):
        for j in range(n-i-1):
            #if L[j]>L[j+1]:
            if hamSoSanh2MangFlatten(L[j],L[j+1]):
                L[j],L[j+1]=L[j+1],L[j]

if __name__ == '__main__':
    #python plot_kernel.py angle 0.2

    #each (type,delta) => list cac kernel ung voi cac L
    #chi xet type : angle va vortex
    #eg: (0.2,xy) => (12,16,24,32)*8 cac kernel
    
    args=sys.argv
    
    type=args[1]
    delta=args[2]
    index_kernel=0 #0 va 1 la thu tu cua CNN thu 1 va thu 2
    number_filter=8 #CNN1 co 8 filter, CNN2 co 16 filter
    list_L=[12,16,24,32]
    #list_L=[12,16]
    #tim v_max va v_min
    #muc dich cho tat ca cac kernel cung 1 scale
    
    v_min=0
    v_max=0
    for L in list_L:
        model_filename='/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/model.h5'%(type,delta,L)
        model=keras.models.load_model(model_filename)
        
        A=model.layers[index_kernel].get_weights()
        B=A[0] #shape (3,3,1,8)

        if np.amax(B)>v_max:
            v_max=np.amax(B)
        if np.amin(B)<v_min:
            v_min=np.amin(B)

    print('v_min',v_min)
    print('v_max',v_max)
    
    
    
    
    dem_figure=0
    for L in list_L:
        dem_figure+=1
        #plt.figure(dem_figure)

        model_filename='/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/model.h5'%(type,delta,L)
        model=keras.models.load_model(model_filename)
        
        A=model.layers[index_kernel].get_weights()
        B=A[0] #shape (3,3,1,8)
        #print('B.shape',B.shape)
        #input('nghiahgs')
        list_kernel=[]
        #index_filter=0
        
        for index_filter in range(number_filter):
            kernel_matrix=B[:,:,:,index_filter]
            kernel_matrix.shape=kernel_matrix.shape[0],kernel_matrix.shape[1] #reshape => 3x3
            #print(kernel_matrix)
            list_kernel.append(kernel_matrix)
        #sap xep list_kernel
        bubble_sort_list(list_kernel)
        
        #v_min=np.amin(list_kernel)
        #v_max=np.amax(list_kernel)
        #print('v_min',v_min)
        #print('v_max',v_max)

        #ve number_filter tren cung 1 hang ngang
        for index_filter in range(number_filter):
            #plt.subplot(dem_figure,number_filter,index_filter+1)
            plt.subplot(len(list_L),number_filter,index_filter+1+(dem_figure-1)*number_filter)
            if index_filter==number_filter-1:
                sns.heatmap(list_kernel[index_filter],vmin=v_min,vmax=v_max)
            else:
                sns.heatmap(list_kernel[index_filter],cbar=False,vmin=v_min,vmax=v_max)
        #plt.show()
        #plt.savefig()

    #plt.show()
    plt.savefig('/home/nghia/generalized_xy/codes/createModel/plot_kernel_result/%s/delta%s/CNN0.pdf'%(type,delta))
