import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils
import pickle
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
    #chu y can thay doi khi la xy hoac angle
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


    L_predict=model2.predict(x_data)
    return L_predict
    #return L_predict[:,positionNeural] #neural first

if __name__ == '__main__':
    args=sys.argv
    type=args[1] #vortex
    data_dir=args[2] #/home/nghia/generalized_xy/delta0.2/hdf5/Delta0.2/L32
    model_dir=args[3] #/home/nghia/generalized_xy/codes/model/vortex/Delta0.2/L12_16_24_32
    
    delta=float(data_dir.split('/')[-2].replace('Delta',''))
    print('delta=',delta)
    tmp = utils.load_data('%s/run20.h5' % data_dir)
    L = np.size(tmp[0], 1)
    
    angle = tmp[1] / (2*np.pi)
    vortex = tmp[2].reshape(-1, L, L, 1)
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
    
    if type=='vortex':
        print('x_data la vortex')
        x_data=vortex #da reshape o tren
    elif type=='angle':
        print('x_data la angle')
        x_data=angle.reshape(-1,L,L,1)
    elif type=='angle_vortex':
        print('x_data la angle_vortex')
        x_data=angle_vortex
    elif type=='xy':
        print('x_data la xy')
        x_data=xy
    else:
        print('else')
    print("x_data.shape",x_data.shape)
    
    #predict_32=cacl_predict_in_layer32('%s/model_lientiep.h5'%model_dir,x_data)
    #predict_32=cacl_predict_in_layer32('%s/model_kernel_5x5.h5'%model_dir,x_data)
    predict_32=cacl_predict_in_layer32('%s/model.h5'%model_dir,x_data)
    #model = keras.models.load_model('%s/model_lientiep.h5'%model_dir)
    #model = keras.models.load_model('%s/model_kernel_5x5.h5'%model_dir)
    model = keras.models.load_model('%s/model.h5'%model_dir)
    predict_softmax = model.predict(x_data)

    #luu lai ket qua evaluate trong 1 file picke cung voi model
    kqsave=(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax )
    #with open('%s/%s_lientiep.pickle'%(model_dir,'_'.join(data_dir.split('/')[-2:])),'wb') as f:
    #with open('%s/%s_kernel_5x5.pickle'%(model_dir,'_'.join(data_dir.split('/')[-2:])),'wb') as f:
    with open('%s/%s.pickle'%(model_dir,'_'.join(data_dir.split('/')[-2:])),'wb') as f:
        pickle.dump(kqsave,f)
    del tmp



