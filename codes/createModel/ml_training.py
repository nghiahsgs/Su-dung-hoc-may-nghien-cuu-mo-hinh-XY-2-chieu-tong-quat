import utils
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

print('Tensorflow versionzz %s' % tf.__version__)

def obtain_train_val_data(angle, vortex, label, temperature,angle_vortex,xy):
    #split data thanh 2 phan train & val
    #angle_vortex : chuan hoa trc khi cho vao
    anorm = 2*np.pi
    
    ind = np.arange(len(label))
    i_train, i_val, l_train, l_val = train_test_split(
            ind, label, test_size=0.3, random_state=243)
    
    train_data = (angle[i_train]/anorm, vortex[i_train],
                  label[i_train], temperature[i_train],
                  angle_vortex[i_train],xy[i_train])
    
    validation_data = (angle[i_val]/anorm, vortex[i_val],
                       label[i_val],temperature[i_val],
                       angle_vortex[i_val],xy[i_val])
    return train_data, validation_data

def create_neural_network_model(L, num_classes, input_shape):
    #input shape theo input
    model = keras.Sequential()
    model.add(layers.Conv2D(8, kernel_size=(3, 3), activation='relu',
			    input_shape=input_shape))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def save(model, progress,result_dir):
    #/home/nghia/generalized_xy/codes/model
    #model.save('%s/model_kernel_5x5.h5'%result_dir)
    #f = h5py.File('%s/progress_kernel_5x5.h5'%result_dir, 'w')
    
    model.save('%s/model.h5'%result_dir)
    f = h5py.File('%s/progress.h5'%result_dir, 'w')

    f.create_dataset('loss', data=progress.history['loss'])
    f.create_dataset('acc', data=progress.history['acc'])
    f.create_dataset('val_loss', data=progress.history['val_loss'])
    f.create_dataset('val_acc', data=progress.history['val_acc'])


if __name__ == '__main__':
    args=sys.argv
    
    type=args[1] #angle
    data_dir=args[2]
    result_dir=args[3]


    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    #c1:load data theo cach cu (nhuoc diem la ton nhieu bo nho|uu diem la tong quat)
    '''
    list_data_file = ['%s/run%02d.h5' % (data_dir, n) for n in range(1, 20)]
    angle, vortex, label, temperature, angle_vortex,xy = utils.load_all_data(list_data_file)
    
    #chia data thanh 2 phan train va val
    train_data, validation_data = obtain_train_val_data(angle, vortex, label,
                                                        temperature,angle_vortex,xy)
    L = np.size(angle, 2)
    
    y_train = train_data[2]
    y_val = validation_data[2]
    if type=="vortex":
        x_train = train_data[1].reshape(-1, L, L, 1)                    
        x_val = validation_data[1].reshape(-1, L, L, 1)
    elif type=="angle":
        x_train = train_data[0].reshape(-1, L, L, 1)                    
        x_val = validation_data[0].reshape(-1, L, L, 1)
    elif type=="angle_vortex": 
        #angle_vortex
        x_train = train_data[4].reshape(-1, L, L, 2)                    
        x_val = validation_data[4].reshape(-1, L, L, 2)
    elif type=="xy":   
        x_train = train_data[5].reshape(-1, L, L, 2)                    
        x_val = validation_data[5].reshape(-1, L, L, 2)
    else:
        print('else')
    '''
    #c2:load data theo cach moi (nhuoc diem la nhieu truong hop, uu diem la tiet kiem bo nho)
    delta=float(data_dir.split('/')[-2].replace('Delta',''))
    L=int(data_dir.split('/')[-1].replace('L',''))
    print('delta=',delta)
    print('L=',L)

    #load run 1 den 19 lam x_train
    list_data_file=['%s/run%02d.h5' % (data_dir,n) for n in range(1,20)]
    #tmp=utils.load_data_more(list_data_file,['angle'])
    tmp=utils.load_data_more(list_data_file,[type])

    xy=tmp[0]
    angle = tmp[1]
    vortex = tmp[2]
    label = tmp[3]
    temperature = tmp[4]
    angle_vortex=tmp[6]
    
    #chuan hoa data
    try:
        angle = angle / (2*np.pi)
    except:
        print('error because of None')

    #reshape data
    try:
        angle=angle.reshape(-1,L,L,1)
    except:
        print('error due reshape fail')

    try:
        vortex = vortex.reshape(-1, L, L, 1)
    except:
        print('error due reshape fail')
    
    try:
        angle_vortex=angle_vortex.reshape(-1,L,L,2)
    except:
        print('error due reshape fail')
    
    try:
        xy=xy.reshape(-1,L,L,2)
    except:
        print('error due reshape fail')
    
    #gan du lieu cho x_data
    if type=='vortex':
        x_data=vortex
    elif type=='angle':
        x_data=angle
    elif type=='angle_vortex':
        x_data=angle_vortex
    elif type=='xy':
        x_data=xy
    else:
        print('else')

    #print('x_data',x_data.shape)
    ind = np.arange(len(label))
    i_train, i_val, l_train, l_val = train_test_split(
            ind, label, test_size=0.3, random_state=243)

    x_train=x_data[i_train]
    x_val=x_data[i_val]
    y_train=label[i_train]
    y_val=label[i_val]
    '''
    print('x_train.shape',x_train.shape)
    print('x_val.shape',x_val.shape)
    print('y_train.shape',y_train.shape)
    print('y_val.shape',y_val.shape)
    input('nghiahsgs')
    '''
    #bat dau train model 
    epochs = 300
    batch_size = 64
    
    num_categories = np.size(label, 1)
    model = create_neural_network_model(L, num_categories, x_train.shape[1:])
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=30)
    progress = model.fit(x_train, y_train, epochs=epochs, batch_size=64,
                         validation_data=(x_val, y_val),
                         callbacks=[early_stopping], verbose=1)
    save(model, progress,result_dir)
