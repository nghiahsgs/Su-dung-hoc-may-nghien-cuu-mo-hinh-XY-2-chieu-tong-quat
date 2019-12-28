import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils
        
def calc_helicity(X_data,T,delta):
    #tinh cho 1 diem #input la angle
    X_data=X_data*2*np.pi
    L=X_data.shape[0]
    
    #tinh helicity x (dich cot tu phai qua trai)
    X_data_shift=np.c_[X_data[:,1:],X_data[:,0]]
    X_data_ij=X_data[:, :] - X_data_shift


    sh1=delta*np.cos(X_data_ij) + 4*(1-delta)*np.cos(2*X_data_ij)
    sh1=np.sum(sh1)
    sh1=sh1/(L*L)

    sh2=delta*np.sin(X_data_ij) + 2*(1-delta)*np.sin(2*X_data_ij)
    sh2=np.sum(sh2)
    sh2=(sh2*sh2)/(T*L*L)

    helicity_x=sh1-sh2
    
    #tinh helicity_y (cac hang lui xuong thap)
    lastRow=X_data[-1,:]
    lastRow.shape=(1,L)
    
    X_data_shift=np.r_[lastRow,X_data[0:-1,:]]
    X_data_ij=X_data[:, :] - X_data_shift

    sh1=delta*np.cos(X_data_ij) + 4*(1-delta)*np.cos(2*X_data_ij)
    sh1=np.sum(sh1)
    sh1=sh1/(L*L)

    sh2=delta*np.sin(X_data_ij) + 2*(1-delta)*np.sin(2*X_data_ij)
    sh2=np.sum(sh2)
    sh2=(sh2*sh2)/(T*L*L)

    helicity_y=sh1-sh2
    
    
    return (helicity_x+helicity_y)/2

def calc_magnetization(X_data):
    #tinh cho 1 diem #X_data: angle
    X_data=X_data*2*np.pi
    L=X_data.shape[0]
   
    #tong spin
    return np.sqrt(np.sum(np.cos(X_data))**2+np.sum(np.sin(X_data))**2)/(L**2)


def convert_vortex_2_sum(vortex):
    sum_vortex=np.zeros(vortex.shape[0],dtype=float)
    L=vortex.shape[1]
    pos=0
    for v in vortex:
        sum_vortex[pos]=np.sum(np.abs(v))/(L*L)
        pos+=1
    return sum_vortex
def cacl_interpolation_2_line(x,y1,y2):
    y3=y2-y1
    plt.figure(2)
    plt.plot(x,y1,'x')
    plt.plot(x,y2,'o')

    #noi suy
    tck1=interpolate.splrep(x,y1,s=0)
    tck2=interpolate.splrep(x,y2,s=0)
    tck3=interpolate.splrep(x,y3,s=0)

    return interpolate.sproot(tck3)

def cacl_predict_in_layer32(model_filename,x_data):
    model=keras.models.load_model(model_filename)
   
    L=x_data.shape[1]
    num_classes = 3
    img_rows, img_cols = L,L
    input_shape = (img_rows, img_cols,1)

    model2 = keras.Sequential()
    model2.add(layers.Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=input_shape
                            ,weights=model.layers[0].get_weights()))
    model2.add(layers.Conv2D(16, (3, 3), activation='relu'
                            ,weights=model.layers[1].get_weights()))
    model2.add(layers.MaxPooling2D(pool_size=(2, 2)
                            ,weights=model.layers[2].get_weights()))
    model2.add(layers.Flatten(
                            weights=model.layers[3].get_weights()))
    model2.add(layers.Dense(32, activation='relu'
                            ,weights=model.layers[4].get_weights()))
    # model2.add(layers.Dense(num_classes, activation='softmax'
    #                        ,weights=model.layers[5].get_weights())) #drop last layer


    L_predict=model2.predict(x_data)
    #print(L_predict)
    return L_predict
    #return L_predict[:,positionNeural] #neural first
def plot_tuong_quan_32neuron_and_dlvl(model_filename,temp_specific,x_data,temp,mag,axes,helicity,icot,ipagepdf):
    x_data_specific=x_data[abs(temp-temp_specific)<1e-8]
    L_predict_32=cacl_predict_in_layer32(model_filename,x_data_specific)
    #print(L_predict_32[:,0])
    
    mag=mag.reshape(-1,1)
    helicity=helicity.reshape(-1,1)

    mag_spe=mag[abs(temp-temp_specific)<1e-8]
    helicity_spe=helicity[abs(temp-temp_specific)<1e-8]
    #ax.plot()
    #print(L_predict_32[:,0].shape)
    #print(mag_spe.shape)
    #for i in range(32):
    for j in range(4):
        i=j+ipagepdf*4
        #print('i',i)
        axes[j,icot].scatter(L_predict_32[:,i],mag_spe)
        axes[j,icot+3].scatter(L_predict_32[:,i],helicity_spe)

def plot_tuong_quan_32neuron_and_dlvl_Alam(model_filename,Tc_pre,Tc,Tc_after,x_data,temp,mag,axes,helicity,ipagepdf):
    L_predict_32=cacl_predict_in_layer32(model_filename,x_data)
    L_predict_32_Tc_pre=L_predict_32[abs(temp-Tc_pre)<1e-8]
    L_predict_32_Tc=L_predict_32[abs(temp-Tc)<1e-8]
    L_predict_32_Tc_after=L_predict_32[abs(temp-Tc_after)<1e-8]
    
    mag=mag.reshape(-1,1)
    helicity=helicity.reshape(-1,1)

    mag_Tc_pre=mag[abs(temp-Tc_pre)<1e-8]
    mag_Tc=mag[abs(temp-Tc)<1e-8]
    mag_Tc_after=mag[abs(temp-Tc_after)<1e-8]
    
    helicity_Tc_pre=helicity[abs(temp-Tc_pre)<1e-8]
    helicity_Tc=helicity[abs(temp-Tc)<1e-8]
    helicity_Tc_after=helicity[abs(temp-Tc_after)<1e-8]
   
    for offset in range(4):

        iNeuron=ipagepdf*4+offset
        print('iNeuron',iNeuron)
        axes[offset,0].scatter(L_predict_32_Tc_pre[:,iNeuron],mag_Tc_pre,c='red')
        axes[offset,0].scatter(L_predict_32_Tc[:,iNeuron],mag_Tc,c='green')
        axes[offset,0].scatter(L_predict_32_Tc_after[:,iNeuron],mag_Tc_after,c='blue')
        
        axes[offset,1].scatter(L_predict_32_Tc_pre[:,iNeuron],helicity_Tc_pre,c='red')
        axes[offset,1].scatter(L_predict_32_Tc[:,iNeuron],helicity_Tc,c='green')
        axes[offset,1].scatter(L_predict_32_Tc_after[:,iNeuron],helicity_Tc_after,c='blue')
       # axes[j,icot+3].scatter(L_predict_32[:,i],helicity_spe)
def plot_prediction(model_filename,x_data,label,temp,mag,helicity):
    print('load plot prediction')
    model = keras.models.load_model(model_filename)
    print('end load plot prediction')
    predict = model.predict(x_data)
    
    ''' => ve trung binh voetex theo nhiet do
    predict=convert_vortex_2_sum(x_data)
    predict=predict.reshape(-1,1)
    '''
    ''' => ve trung binh mag theo nhiet do
    predict=mag
    predict=predict.reshape(-1,1)
    '''
    
    ''' => ve trung binh helicity theo nhiet do
    predict=helicity
    predict=predict.reshape(-1,1)
    '''

    #ve trung binh predict (theo nhiet do) theo nhiet do trung binh
    temp_set = np.array(list(set(temp)))
    out = np.zeros((len(temp_set), np.size(predict, 1)))
    for n, T in enumerate(temp_set):
        out[n] = np.mean(predict[abs(temp - T) < 1e-8], 0)
    ind = np.argsort(temp_set)
    ''' => chi ve trung binh dai luong theo nhiet do
    plt.plot(temp_set[ind], out[ind])
    plt.show()
    '''
    #''' => ve va tinh noi suy 3 duong
    x=temp_set[ind]
    y1=out[ind,0]
    y2=out[ind,2]
    T_cross=cacl_interpolation_2_line(x,y1,y2)
    print(T_cross)
    
    plt.figure(1)
    plt.plot(temp_set[ind], out[ind,0])
    plt.plot(temp_set[ind], out[ind,1])
    plt.plot(temp_set[ind], out[ind,2])
    #plt.legend(['P','N','F'],loc='upper left')
    plt.legend(['P','N','F'])
    plt.show()
    #'''
if __name__ == '__main__':
    args=sys.argv
    
    type=args[1]
    data_dir=args[2]
    model_dir=args[3]
    
    #data_dir = '/home/nghia/generalized_xy/delta1.0/hdf5/Delta1.0/L48'
    delta=0.34
    tmp = utils.load_data('%s/run20.h5' % data_dir)
    L = np.size(tmp[0], 1)
    
    angle = tmp[1] / (2*np.pi)
    vortex = tmp[2].reshape(-1, L, L, 1)
    #x_data = np.c_[angle.reshape(-1, L, L, 1), vortex]
    label = tmp[3]
    temp = tmp[4]
    angle_vortex=tmp[6]
    xy=tmp[0]
   
    #tinh toan them cac dai luong vat ly
    helicity=np.zeros([angle.shape[0]],dtype=float)
    mag=np.zeros([angle.shape[0]],dtype=float)
    pos=0
    for i in range(angle.shape[0]):
        helicity[pos]=calc_helicity(angle[i],temp[i],delta)
        mag[pos]=calc_magnetization(angle[i])
        pos+=1
    #plt.plot(temp,mag)
    #plot lun
    #plot_prediction('model.h5',x_data,label,temp)
    if type=='vortex':
        x_data=vortex #da reshape o tren
    elif type=='angle':
        x_data=angle.reshape(-1,L,L,1)
    elif type=='angle_vortex':
        x_data=angle_vortex
    elif type=='xy':
        x_data=xy
    else:
        print('else')
    plot_prediction('%s/model.h5'%model_dir,x_data,label,temp,mag,helicity)
    input('zzz')
    #print(sorted(set(temp)))
    #'''
    for ipagepdf in range(8):
        '''
        fig, axes = plt.subplots(4, 6,figsize=(7*4,7*6))
        #truoc chuyen pha
        plot_tuong_quan_32neuron_and_dlvl('model.h5',0.8700000000000001,vortex,temp,mag,axes,helicity,0,ipagepdf)
        #chuyen pha
        plot_tuong_quan_32neuron_and_dlvl('model.h5',0.8940000000000001,vortex,temp,mag,axes,helicity,1,ipagepdf)
        #sau chuyen pha
        plot_tuong_quan_32neuron_and_dlvl('model.h5',0.9180000000000001,vortex,temp,mag,axes,helicity,2,ipagepdf)
        fig.savefig('test'+str(ipagepdf)+'.pdf')
        print('done graphic')
        '''

        fig, axes = plt.subplots(4, 2,figsize=(7*4,7*2))
        plot_tuong_quan_32neuron_and_dlvl_Alam('model.h5',0.87,0.894,0.918,vortex,temp,mag,axes,helicity,ipagepdf)
        fig.savefig('test'+str(ipagepdf)+'.pdf')
        print('done graphic')
    #'''
    del tmp


