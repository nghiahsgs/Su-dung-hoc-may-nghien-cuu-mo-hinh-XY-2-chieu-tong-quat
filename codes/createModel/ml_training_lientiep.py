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
def create_model_from_old_model(old_model,L,num_classes,input_shape):
    
    #model=keras.models.load_model(model_filename)
    model=old_model
    
    #L=x_data.shape[1]
    #num_classes = 3
    #img_rows, img_cols = L,L
    #input_shape = (img_rows, img_cols,1)

    model2 = keras.Sequential()
    model2.add(layers.Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=input_shape,weights=model.layers[0].get_weights()))
    model2.add(layers.Conv2D(16, (3, 3), activation='relu',weights=model.layers[1].get_weights()))
    model2.add(layers.MaxPooling2D(pool_size=(2, 2),weights=model.layers[2].get_weights()))
    model2.add(layers.Flatten(weights=model.layers[3].get_weights()))
    
    matrix_flat=model.layers[4].get_weights()
    #print('miatrix_falt',matrix_flat[0].shape)
    if L==16:
        A=matrix_flat[0]
        B=np.random.uniform(-0.05,0.05,(576,32))
        pad=(576-256)//2

        C=np.r_[B[0:pad][:],A,B[-pad:][:]]
        matrix_flat[0]=C
    if L==24:
        A=matrix_flat[0]
        B=np.random.uniform(-0.05,0.05,(1600,32))
        pad=(1600-576)//2

        C=np.r_[B[0:pad][:],A,B[-pad:][:]]
        matrix_flat[0]=C
    if L==32:
        A=matrix_flat[0]
        B=np.random.uniform(-0.05,0.05,(3136,32))
        pad=(3136-1600)//2

        C=np.r_[B[0:pad][:],A,B[-pad:][:]]
        matrix_flat[0]=C

    
    #model2.add(layers.Dense(32, activation='relu',weights=model.layers[4].get_weights()))
    model2.add(layers.Dense(32, activation='relu',weights=matrix_flat))
    #model2.add(layers.Dense(32, activation='relu'))#do bi flatten nen vs L khac nhau se khac
    
    model2.add(layers.Dense(num_classes, activation='softmax',weights=model.layers[5].get_weights()))

    model2.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model2
def save(model, progress,result_dir):
    #/home/nghia/generalized_xy/codes/model
    model.save('%s/model_lientiep.h5'%result_dir)
    f = h5py.File('%s/progress_lieptiep.h5'%result_dir, 'w')
    f.create_dataset('loss', data=progress.history['loss'])
    f.create_dataset('acc', data=progress.history['acc'])
    f.create_dataset('val_loss', data=progress.history['val_loss'])
    f.create_dataset('val_acc', data=progress.history['val_acc'])


if __name__ == '__main__':
    #muc dich: cung 1 type, cung 1 delta => quet qua tat ca Ls[12,16,24,32]
    
    args=sys.argv
    
    type=args[1] #angle
    delta=args[2] #0.2
    list_L=[12,16,24,32]
    #list_L=[12,24,16,32]
    #list_L=[12,16] 
    for L in list_L:

        data_dir = '/home/nghia/generalized_xy/delta%s/hdf5/Delta%s/L%s'%(delta,delta,L)
        #list_data_file = ['%s/run%02d.h5' % (data_dir, n) for n in range(1, 2)]
        list_data_file = ['%s/run%02d.h5' % (data_dir, n) for n in range(1, 20)]
    
        #result_dir="/home/nghia/generalized_xy/codes/model/%s/Delta%s/L12_16_24_32"%(type,delta)
        result_dir="/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s"%(type,delta,L)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

    
        #first run
        angle, vortex, label, temperature, angle_vortex,xy = utils.load_all_data(list_data_file)
        #sys.exit() #dung de xem co gia tri vorext nao bang 1/2 ko
    
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
    
        epochs = 300
        batch_size = 64
    
        num_categories = np.size(label, 1)
        #khoi tao model
        print('L=%s'%L)
        if L==12:
            model = create_neural_network_model(L, num_categories, x_train.shape[1:])
        else:
            old_model=model
            model=create_model_from_old_model(old_model,L, num_categories, x_train.shape[1:])
  #          input('nghiahsgs')

         
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)

        #train model
        progress = model.fit(x_train, y_train, epochs=epochs, batch_size=64,
                         validation_data=(x_val, y_val),
                         callbacks=[early_stopping], verbose=1)
        #luu ket qua sau cung
        save(model, progress,result_dir)
        
