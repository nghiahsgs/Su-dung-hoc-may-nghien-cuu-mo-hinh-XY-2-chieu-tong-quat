import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils
import pickle
def cacl_interpolation_2_line(x,y1,y2):
    y3=y2-y1
    #plt.figure(2)
    #plt.plot(x,y1,'x')
    #plt.plot(x,y2,'o')

    #noi suy
    tck3=interpolate.splrep(x,y3,s=0)
    return interpolate.sproot(tck3)
                                    
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
def plot_prediction(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax,data_dir):
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

    predict=predict_softmax
    #ve trung binh predict (theo nhiet do) theo nhiet do trung binh
    
    temp_set = np.array(list(set(temp)))
    out = np.zeros((len(temp_set), np.size(predict, 1)))
    for n, T in enumerate(temp_set):
        out[n] = np.mean(predict[abs(temp - T) < 1e-8], 0)
    ind = np.argsort(temp_set)
    
    #''' => chi ve trung binh dai luong theo nhiet do
    print(temp_set[ind].shape)
    print(out[ind].shape)
    plt.plot(temp_set[ind], out[ind])
    
    data_dir='/'.join(data_dir.split('/')[:-1])
    #data_dir="%s/lientiep.pdf"%data_dir
    data_dir="%s/PvsT.pdf"%data_dir

    print('data_dir',data_dir)
    #plt.savefig('lientiep.pdf')
    plt.savefig(data_dir)
    plt.show()
    print('da ve xong')
    #'''
    ''' => ve va tinh noi suy 3 duong
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
    '''
def plot_activation_32(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmaxi,i_neuron,Tc_before,Tc_after,isBefore):
    if isBefore:
        predict_32_Tc_before=predict_32[abs(temp-Tc_before)<1e-8]
    else:
        predict_32_Tc_before=predict_32[abs(temp-Tc_after)<1e-8]
    
    
    predict=predict_32_Tc_before[:,i_neuron]
    

    plt.plot(np.zeros(len(predict),dtype='int32'),predict,'x') 
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False)
    #plt.show()
def calc_Tc(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax):
    predict=predict_softmax
    #ve trung binh predict (theo nhiet do) theo nhiet do trung binh
    
    temp_set = np.array(list(set(temp)))
    out = np.zeros((len(temp_set), np.size(predict, 1)))
    for n, T in enumerate(temp_set):
        out[n] = np.mean(predict[abs(temp - T) < 1e-8], 0)
    ind = np.argsort(temp_set)
    
    #''' => chi ve trung binh dai luong theo nhiet do
    #plt.plot(temp_set[ind], out[ind])
    #plt.show()
    #'''
    # => ve va tinh noi suy 3 duong
    x=temp_set[ind]
    y1=out[ind,0]
    y2=out[ind,2]
    T_cross=cacl_interpolation_2_line(x,y1,y2)
    #print(T_cross)
    return T_cross
    
if __name__ == '__main__':
    #check xem co vortex nao =1/2 ko
    '''data_dir = '/home/nghia/generalized_xy/codes/model/vortex/Delta0.2/L12_16_24_32/Delta0.2_L32.pickle'
    with open(data_dir,'rb') as f:
        kqsave=pickle.load(f)

    xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave                                                                                                                                                                                                                                                                                                             
    sys.exit()
    '''
    #chuc nang 1: plot P vs T theo (type,delta,L)
    '''
    args=sys.argv
    type=args[1]
    delta=args[2]
    L=args[3]

    #data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s_lientiep.pickle'%(type,delta,L,delta,L)
    #data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s_kernel_5x5.pickle'%(type,delta,L,delta,L)
    data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
    print(data_dir)
    with open(data_dir,'rb') as f:
        kqsave=pickle.load(f)

    
    xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
    #print('predict_softmax',predict_softmax.shape)

    plot_prediction(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax,data_dir)
    
    #input('end chuc nang 1')
    sys.exit()
    '''

    #chuc nang 2: plot P vs T theo (delta,L) : ve cung cac loai tren cung 1 hinh
    '''
    list_type=['xy','angle','vortex','angle_vortex']
    
    args=sys.argv
    delta=args[1]#0.34
    L=args[2]#12
    
    #data_dir = '/home/nghia/generalized_xy/codes/model/xy/Delta0.34/L12/Delta0.34_L12.pickle'
    dem=0
    #plt.figure(figsize=(6,3))
    
    for type in list_type:
        dem+=1
        plt.subplot(1,5,dem)
        plt.xlabel('%s,%s,%s'%(delta,L,type))
        data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
        print(data_dir)
        with open(data_dir,'rb') as f:
            kqsave=pickle.load(f)

        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
        plot_prediction(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
    

    plt.subplot(1,5,5)
    for type in list_type:
        plt.xlabel('%s,%s,%s'%(delta,L,'combined'))
        data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
        print(data_dir)
        with open(data_dir,'rb') as f:
            kqsave=pickle.load(f)

        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
        plot_prediction(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
    plt.savefig('resultPvsT/delta%s_L%s.pdf'%(delta,L))
    sys.exit()
    input("xong buoc 2")
    #plt.show()
    '''
    #chuc nang 3: plot P vs T theo (delta,type) : ve cung cac loai tren cung 1 hinh
    '''
    list_L=[12,16,24,32]
    
    args=sys.argv
    delta=args[1]#0.34
    type=args[2]#xy
    
    #data_dir = '/home/nghia/generalized_xy/codes/model/xy/Delta0.34/L12/Delta0.34_L12.pickle'
    dem=0
    #plt.figure(figsize=(6,3))
    
    for L in list_L:
        dem+=1
        plt.subplot(1,5,dem)
        plt.xlabel('%s,%s,%s'%(delta,L,type))
        data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
        print(data_dir)
        with open(data_dir,'rb') as f:
            kqsave=pickle.load(f)

        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
        plot_prediction(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
    

    plt.subplot(1,5,5)
    for L in list_L:
        plt.xlabel('%s,%s,%s'%(delta,L,'combined'))
        data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
        print(data_dir)
        with open(data_dir,'rb') as f:
            kqsave=pickle.load(f)

        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
        plot_prediction(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
    plt.savefig('resultPvsT/delta%s_type%s.pdf'%(delta,type))
    sys.exit()
    #plt.show()
    '''
    #chuc nang 4: luu tat cac cac diem cat nhau Tc ra file
    '''
    list_L=[12,16,24,32]
    list_type=['xy','angle','vortex','angle_vortex']
    list_delta=['0.2','0.34','1.0']
    data_Tc=np.empty((len(list_L),len(list_type),len(list_delta)))
    
    for i_L in range(len(list_L)):
        L=list_L[i_L]
        for i_type in range(len(list_type)):
            type=list_type[i_type]
            for i_delta in range(len(list_delta)):
                delta=list_delta[i_delta]
                data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
                print(data_dir)
                with open(data_dir,'rb') as f:
                    kqsave=pickle.load(f)

                xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
                Tc=calc_Tc(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
                Tc=Tc[0]
                print(Tc)
                #input("nghiahsgs")
                data_Tc[i_L][i_type][i_delta]=Tc
    print(data_Tc)
    np.save('resultPvsT/data_Tc',data_Tc)
    sys.exit()
    '''
    #chuc nang 5: ve activation_32 va dem so neuron hoat dong
    '''
    list_L=[12,16,24,32]
    list_type=['xy','angle','vortex','angle_vortex']
    list_delta=['0.34','1.0']
    data_Tc=np.load("resultPvsT/data_Tc.npy")
    
    dem_figure=0
    for i_L in range(len(list_L)):
        L=list_L[i_L]
        for i_type in range(len(list_type)):
            type=list_type[i_type]
            for i_delta in range(len(list_delta)):
                dem_figure+=1
                plt.figure(dem_figure)

                delta=list_delta[i_delta]
                data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
                print(data_dir)
                with open(data_dir,'rb') as f:
                    kqsave=pickle.load(f)

                xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
                print("temp.shape",temp.shape)
                print("predict_32.shape",predict_32.shape)
                
                Tc=data_Tc[i_L,i_type,i_delta]
                print('Tc',Tc)
                
                array_temp=np.sort(np.array(list(set(temp))))
                Tc_before=array_temp[array_temp<Tc][-2]
                Tc_after=array_temp[array_temp>Tc][1]

                #print('Tc_before',Tc_before)
                #print('Tc_after',Tc_after)
                #print('array_temp',array_temp)

                i_neuron=0
                
                dem=0
                
                for i_neuron in range(32):
                    dem+=1
                    plt.subplot(1,32,dem)
                    plt.xlabel("%s"%i_neuron)
                    plot_activation_32(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax,i_neuron,Tc_before,Tc_after)
                plt.savefig('resultNeuronLazy/L%s_type%s_delta%s_after.pdf'%(L,type,delta))
                #plt.show()
                #input("nghiahsgs")
    #plt.show() 
    '''
    #chuc nang 6: Ve activation function cua tung model 1 => dem bn neural hoat dong
    '''
    type="xy"
    delta="1.0"
    L=12
    isBefore=False

    data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
    #data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s_lientiep.pickle'%(type,delta,L,delta,L)
    print(data_dir)
    with open(data_dir,'rb') as f:
        kqsave=pickle.load(f)

        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
        print("temp.shape",temp.shape)
        print("predict_32.shape",predict_32.shape)
                
    Tc=calc_Tc(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
    Tc=Tc[0]
    
    array_temp=np.sort(np.array(list(set(temp))))
    Tc_before=array_temp[array_temp<Tc][-2]
    Tc_after=array_temp[array_temp>Tc][1]
    print(Tc_before)
    print(Tc)
    print(Tc_after)


    dem=0
                
    for i_neuron in range(32):
        dem+=1
        plt.subplot(1,32,dem)
        plt.xlabel("%s"%i_neuron)
        plot_activation_32(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax,i_neuron,Tc_before,Tc_after,isBefore)
    #plt.savefig('resultNeuronLazy/L%s_type%s_delta%s_after.pdf'%(L,type,delta))
    #plt.savefig('32.pdf')
    
    data_dir='/'.join(data_dir.split('/')[:-1])
    if isBefore:
        #data_dir="%s/lk_%s_%s_%s_before.pdf"%(data_dir,L,delta,type)
        data_dir="%s/%s_%s_%s_before.pdf"%(data_dir,L,delta,type)
    else:
        #data_dir="%s/lk_%s_%s_%s_after.pdf"%(data_dir,L,delta,type)
        data_dir="%s/%s_%s_%s_after.pdf"%(data_dir,L,delta,type)
    print(data_dir)
    plt.savefig(data_dir)
    
    #sys.exit()
    #plt.show()
    '''
    #chuc nang 7:moi neuron: ve mag va helicity theo activation (T<Tc, T=Tc,T>Tc)
    '''
    import os

    args=sys.argv
    type=args[1]
    delta=args[2]
    L=args[3]

    data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
    #data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s_lientiep.pickle'%(type,delta,L,delta,L)
    result_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/neurons'%(type,delta,L)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    if not os.path.isdir('%s/mag'%(result_dir)):
        os.makedirs('%s/mag'%(result_dir))

    if not os.path.isdir('%s/helicity'%(result_dir)):
        os.makedirs('%s/helicity'%(result_dir))


    with open(data_dir,'rb') as f:
        kqsave=pickle.load(f)

        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
        print("temp.shape",temp.shape)
        print("predict_32.shape",predict_32.shape)
                
    Tc=calc_Tc(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
    Tc=Tc[0]
    
    array_temp=np.sort(np.array(list(set(temp))))
    #Tc_before=array_temp[array_temp<Tc][-2]
    #Tc_after=array_temp[array_temp>Tc][1]
    Tc_before=array_temp[1]
    Tc_after=array_temp[-2]
    print(Tc_before)
    print(Tc)
    print(Tc_after)
    #input('nghiahsgs')
    for i in range(32):
        ineuron=i
        plt.figure()
        ax1=plt.subplot(1,2,1)
        index_T_less_Tc=np.where(temp==Tc_before)[0]
        plt.xlabel("activation (neuron %s) of respective points "%ineuron)
        #plt.ylabel("mag (neuron %s) of points which have T<Tc[-2] "%ineuron)
        plt.ylabel("mag (neuron %s) of points which have T=array_temp[1] "%ineuron)
        plt.plot(predict_32[index_T_less_Tc][:,ineuron],mag[index_T_less_Tc],'x')


        plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
        index_T_more_Tc=np.where(temp==Tc_after)[0]
        #plt.xlabel("activation (neuron %s) of respective points"%ineuron)
        #plt.ylabel("mag (neuron %s) of points which have T>Tc[1] "%ineuron)
        plt.plot(predict_32[index_T_more_Tc][:,ineuron],mag[index_T_more_Tc],'x')

        #plt.show()
        plt.savefig('%s/mag/neuron%s.pdf'%(result_dir,ineuron))
    
    for i in range(32):
        ineuron=i
        plt.figure()
        ax1=plt.subplot(1,2,1)
        index_T_less_Tc=np.where(temp==Tc_before)[0]
        plt.xlabel("activation (neuron %s) of respective points "%ineuron)
        plt.ylabel("mag (neuron %s) of points which have T<Tc[-2] and T>Tc[1]"%ineuron)
        plt.plot(predict_32[index_T_less_Tc][:,ineuron],helicity[index_T_less_Tc],'x')


        plt.subplot(1,2,2,sharex=ax1,sharey=ax1)
        index_T_more_Tc=np.where(temp==Tc_after)[0]
        #plt.xlabel("activation (neuron %s) of respective points"%ineuron)
        #plt.ylabel("mag (neuron %s) of points which have T>Tc[1] "%ineuron)
        plt.plot(predict_32[index_T_more_Tc][:,ineuron],helicity[index_T_more_Tc],'x')

        #plt.show()
        plt.savefig('%s/helicity/neuron%s.pdf'%(result_dir,ineuron))
        '''
    #chuc nang 8: ve trung binh activation vs nhiet do
    '''
    import os

    args=sys.argv
    type=args[1]
    delta=args[2]
    L=args[3]

    data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
    result_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/neurons'%(type,delta,L)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    if not os.path.isdir('%s/mag'%(result_dir)):
        os.makedirs('%s/mag'%(result_dir))

    if not os.path.isdir('%s/helicity'%(result_dir)):
        os.makedirs('%s/helicity'%(result_dir))


    with open(data_dir,'rb') as f:
        kqsave=pickle.load(f)

        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
        print("temp.shape",temp.shape)
        print("predict_32.shape",predict_32.shape)
                
    Tc=calc_Tc(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
    Tc=Tc[0]
    
    #predict=predict_32[:,0:5]
    predict=predict_32
    temp_set = np.array(list(set(temp)))
    out = np.zeros((len(temp_set), np.size(predict, 1)))
    for n, T in enumerate(temp_set):
        out[n] = np.mean(predict[abs(temp - T) < 1e-8], 0)
    ind = np.argsort(temp_set)
    
    print(temp_set[ind].shape)
    print(out[ind].shape)
    plt.plot(temp_set[ind], out[ind])

    #np.save('ac_32neuron_0.34_32',np.c_[temp_set[ind],out[ind]])
    #plt.legend(['0','1','2','3','0','1','2','3','0','1','2','3','0','1','2','3','0','1','2','3','0','1','2','3','0','1','2','3','0','1','2','3'])
    plt.savefig('abc.pdf')
    
    input('nghiahsgs')
    plt.savefig('%s/32neurons.pdf'%(result_dir))
    '''
    #chuc nang 9: finite scaling
    '''
    import os

    args=sys.argv
    type=args[1]
    delta=args[2]
    Ls=[12,16,24,32,40,48] #034, angle
    Ls=[12,16,24,32,40,48,56,64] #034, vortex
    Ls=[12,16,24,32,64] #angle, 0.7
    Ls=[12,16,24,32] #angle, 1.0
    
    #create result dir to store pdf plot
    result_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/finite-scaling'%(type,delta,'12')

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    
    for nu in range(1,18):
        nu=nu/10
        print(nu)
        plt.figure()
        plt.xlabel('(T-Tc)/Tc*(L**(1/nu))')
        plt.ylabel('softmax predict')
        #dem=0
        for L in Ls:
            #dem+=1
            #plt.subplot(3,1,dem)
            data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
            print(data_dir)

            with open(data_dir,'rb') as f:
                kqsave=pickle.load(f)

                xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave
                    
            Tc=calc_Tc(xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax)
            Tc=Tc[0]
            
            #ve trung binh xac suat theo nhiet do   
            temp_set = np.array(list(set(temp)))
            out = np.zeros((len(temp_set), np.size(predict_softmax, 1)))
            for n, T in enumerate(temp_set):
                out[n] = np.mean(predict_softmax[abs(temp - T) < 1e-8], 0)
            ind = np.argsort(temp_set)
            #x_old=temp_set[ind]
            x_new=(temp_set[ind]-Tc)/Tc*(L**(1/nu)) 
            plt.plot(x_new, out[ind])
        #plt.legend([12,12,16,16,24,24])
        plt.savefig('%s/finite-scaling-%s.pdf'%(result_dir,nu))
        '''
    #chuc nang 10: tinh slope cua predict trung binh vs T
    #tinh xong he so thi luu lai vao array => save .npy
    '''
    from sklearn.linear_model import LinearRegression 
    args=sys.argv
    type=args[1]
    delta=args[2]
    #L=args[3]
    
    result_dir="/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s"%(type,delta,12)

    array_heso=[]
    #for L in [12,16,24,32,40,48,56,64]:
    for L in [12,16,24,32,64]:
    #for L in [56,64]:

        data_dir = '/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s/Delta%s_L%s.pickle'%(type,delta,L,delta,L)
        print(data_dir)
        with open(data_dir,'rb') as f:
            kqsave=pickle.load(f)

        
        xy,angle,vortex, label,temp,angle_vortex,helicity,mag,predict_32,predict_softmax=kqsave

        
        predict=predict_softmax
        
        temp_set = np.array(list(set(temp)))
        out = np.zeros((len(temp_set), np.size(predict, 1)))
        for n, T in enumerate(temp_set):
            out[n] = np.mean(predict[abs(temp - T) < 1e-8], 0)
        ind = np.argsort(temp_set)
        
        print(temp_set[ind].shape)
        print(out[ind].shape)
        #plt.plot(temp_set[ind], out[ind])
        #plt.savefig('nghiahsgs.pdf')
        #sys.exit()
        
        X=temp_set[ind]
        Y=out[ind][:,0] #dung cho truong hop 1.0 va 0.34 va 0.2 lan 2
        #Y=out[ind][:,2] #dung cho 0.2 lan 1
        

        X_linear= X[np.logical_and(Y>0.2,Y<0.8)] 
        Y_linear= Y[np.logical_and(Y>0.2,Y<0.8)] 
        # danh cho truong hop rat doc,vd 0.34 and 0.2_1
        #if L>=56:
            #X_linear= X[np.logical_and(Y>0.1,Y<0.9)] 
            #Y_linear= Y[np.logical_and(Y>0.1,Y<0.9)] 

            #plt.plot(X,Y,'o')
            #plt.plot(X_linear,Y_linear)
            #plt.savefig('nghiahsgs.pdf')
            #input('aa')
        model=LinearRegression().fit(X_linear.reshape(-1,1),Y_linear)
        print(model.coef_)
        array_heso.append(model.coef_)
    #np.save('%s/array_coef_%s_%s_2'%(result_dir,type,delta),array_heso)
    np.save('%s/array_coef_%s_%s'%(result_dir,type,delta),array_heso)
            
    #plt.plot([12,16,24,32,40,48,56,64],array_heso)
    #plt.savefig('nghiahsgs.pdf')
    sys.exit()
    '''
    #chuc nang 11: load lai cac file r ve he so theo L
    #'''
    import numpy as np
    import matplotlib.pyplot as plt
    
    type='angle'
    

    list_L=[12,16,24,32,40,48,56,64]

    delta=0.2
    result_dir="/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s"%(type,delta,12)
    heso_02_1=np.load('%s/array_coef_%s_0.2_1.npy'%(result_dir,type))
    heso_02_2=np.load('%s/array_coef_%s_0.2_2.npy'%(result_dir,type))
    
    delta=0.34
    result_dir="/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s"%(type,delta,12)
    heso_034=np.load('%s/array_coef_%s_0.34.npy'%(result_dir,type))
    
    delta=0.7
    result_dir="/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s"%(type,delta,12)
    heso_07=np.load('%s/array_coef_%s_0.7.npy'%(result_dir,type))

    delta=1.0
    result_dir="/home/nghia/generalized_xy/codes/model/%s/Delta%s/L%s"%(type,delta,12)
    heso_10=np.load('%s/array_coef_%s_1.0.npy'%(result_dir,type))
    
    plt.figure()
    plt.xlabel('L')
    plt.ylabel('coef slope (P vs T)')
    plt.plot(list_L,-heso_02_1,'.')
    plt.plot(list_L,heso_02_2,'--')
    plt.plot(list_L,heso_034,'x')
    plt.plot([12,16,24,32,64],heso_07,'bo')
    plt.plot(list_L,heso_10,'o')
    plt.legend(['0.2_1','0.2_2','0.34','0.7','1.0'])
    plt.savefig('slope_vs_L.pdf')

    plt.figure()
    plt.xlabel('1/L')
    plt.ylabel('coef slope (P vs T)')
    plt.plot(1/np.array(list_L),-heso_02_1,'.')
    plt.plot(1/np.array(list_L),heso_02_2,'--')
    plt.plot(1/np.array(list_L),heso_034,'x')
    plt.plot(1/np.array([12,16,24,32,64]),heso_07,'bo')
    plt.plot(1/np.array(list_L),heso_10,'o')
    plt.legend(['0.2_1','0.2_2','0.34','0.7','1.0'])
    plt.savefig('slope_vs_1_L.pdf')
    sys.exit()
    #'''
