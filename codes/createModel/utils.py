import h5py
import numpy as np
import matplotlib.pyplot as plt


def convert_xy_to_angle(xy):
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    angle = np.arctan2(y, x)
    angle[y <= 0] += 2*np.pi
    return angle

def saw(x):
    if np.isscalar(x):
        x = np.array([x])
    x[x <= -np.pi] += 2*np.pi
    x[x >= np.pi] -= 2*np.pi
    return x

def calculate_vortex_plaquette(a, print_out=False):
    a1 = a[0, 0]
    a2 = a[0, 1]
    a3 = a[1, 1]
    a4 = a[1, 0]
    if print_out:
        print((a2-a1)/np.pi, saw(a2-a1))
        print((a3-a2)/np.pi, saw(a3-a2))
        print((a4-a3)/np.pi, saw(a4-a3))
        print((a1-a4)/np.pi, saw(a1-a4))
    return (saw(a2-a1)+saw(a3-a2)+saw(a4-a3)+saw(a1-a4))/(2*np.pi)

def calculate_vortex(x):
    x_pbc = np.c_[x, x[:,0]]
    x_pbc = np.r_[x_pbc, [x_pbc[0,:]]]

    x_shifted_left = x_pbc[:-1, 1:]
    x_shifted_up = x_pbc[1:, :-1]
    x_shifted_up_left = x_pbc[1:, 1:]

    s1 = saw(x_shifted_left - x)
    s2 = saw(x_shifted_up_left - x_shifted_left)
    s3 = saw(x_shifted_up - x_shifted_up_left)
    s4 = saw(x - x_shifted_up)
    ret = np.around((s1 + s2 + s3 + s4) / (2*np.pi))
    return np.asarray(ret, dtype=int)

def analyze_vortices(xy):
    angle = convert_xy_to_angle(xy)
    vortex = calculate_vortex(angle)
    print(angle/np.pi)
    print(vortex)
    L = len(angle)
    plt.quiver(np.arange(L), -np.arange(L),
               xy[:, :, 0], xy[:, :, 1], pivot='mid')
    vortex_ref = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if vortex[j, i] > 0:
                plt.plot(i+0.5, -(j+0.5), 'or')
            elif vortex[j, i] < 0:
                plt.plot(i+0.5, -(j+0.5), 'sb')
            if i < L-1 and j < L-1:
                vortex_ref[i, j] = calculate_vortex_plaquette(angle[i:i+2, j:j+2])
    vortex_ref = np.asarray(np.around(vortex_ref), dtype=int)
    print(vortex_ref)
    plt.show()
    return vortex_ref

def get_vortex_density(y_data, vortex):
    L = np.size(vortex, 1)
    temp = set()
    all_temp = np.around(y_data[:, -1], 3)
    ret = None
    for T in all_temp:
        if T in temp:
            continue
        temp.add(T)
        ind = abs(all_temp - T) < 1e-6
        vortex_T = abs(vortex[ind].reshape(-1, L*L))
        vortex_density = np.mean(np.sum(vortex_T, 1) / float(L**2))
        if ret is None:
            ret = [T, vortex_density]
        else:
            ret = np.vstack((ret, [T, vortex_density]))
    return ret
def load_data_more(list_data_file,list_type_load):
    #dem tat ca cac group trong list_data_file
    dem_group=0
    for data_file in list_data_file:
        print('Loading file nho %s' % data_file)
        f = h5py.File(data_file, 'r')
        L = f['L'][()]
        for key in f.keys():
            if not isinstance(f[key], h5py.Group):
                continue
            dem_group+=1
    print('dem_group',dem_group)
    
    #4 loai can load
    xy=None
    angle_vortex=None
    angle=None
    vortex=None

    for type_load in list_type_load:
        if(type_load=="xy"):
            xy = np.zeros([dem_group, L, L, 2], dtype=float)
        elif(type_load=="angle_vortex"):
            angle_vortex = np.zeros([dem_group, L, L, 2], dtype=float)
        elif(type_load=="angle"):
            angle = np.zeros([dem_group, L, L], dtype=float)
        elif(type_load=="vortex"):
            vortex = np.zeros([dem_group, L, L], dtype=int)
   
    label = np.zeros([dem_group, 3], dtype=int)
    temperature = np.zeros([dem_group], dtype=float)

    # get data from HDF5 file
    group_mapping = []
    pos = 0

    for data_file in list_data_file:
        print('Loading file nho %s' % data_file)
        f = h5py.File(data_file, 'r')
        for key in f.keys():
            if not isinstance(f[key], h5py.Group):
                continue
            for type_load in list_type_load:
                if(type_load=="xy"):
                    xy[pos] = f[key]['config'][()]/127.5 - 1.
                elif(type_load=="angle"):
                    xy_temp=f[key]['config'][()]/127.5 - 1.
                    angle[pos] = convert_xy_to_angle(xy_temp)
                elif(type_load=="vortex"):
                    xy_temp=f[key]['config'][()]/127.5 - 1.
                    angle_temp=convert_xy_to_angle(xy_temp)
                    vortex[pos] = calculate_vortex(angle_temp)
                elif(type_load=="angle_vortex"):
                    xy_temp=f[key]['config'][()]/127.5 - 1.
                    angle_temp=convert_xy_to_angle(xy_temp)
                    vortex_temp=calculate_vortex(angle_temp)

                    angle_vortex_each=np.zeros((L,L,2),dtype=float)
                    angle_vortex_each[:,:,0]=angle_temp/(2*np.pi)
                    angle_vortex_each[:,:,1]=vortex_temp
                    angle_vortex[pos]=angle_vortex_each
            
            label[pos, f[key]['label'][()]] = 1
            temperature[pos] = f[key]['T'][()]
            group_mapping.append(key)
            pos += 1
    return xy, angle, vortex, label, temperature, group_mapping,angle_vortex



def load_data(data_file):
    print('Loading file nho %s' % data_file)
    # initialize stuffs
    f = h5py.File(data_file, 'r')
    L = f['L'][()]
    #dem_group = 22800
    dem_group=0
    print(dem_group)
    for key in f.keys():
        if not isinstance(f[key], h5py.Group):
            continue
        dem_group+=1
    
    xy = np.zeros([dem_group, L, L, 2], dtype=float)
    angle_vortex = np.zeros([dem_group, L, L, 2], dtype=float)
    angle = np.zeros([dem_group, L, L], dtype=float)
    vortex = np.zeros([dem_group, L, L], dtype=int)
    label = np.zeros([dem_group, 3], dtype=int)
    temperature = np.zeros([dem_group], dtype=float)

    # get data from HDF5 file
    group_mapping = []
    pos = 0
    for key in f.keys():
        if not isinstance(f[key], h5py.Group):
            continue
#        if pos != dem_group-1:
#            pos += 1
#            continue
        xy[pos] = f[key]['config'][()]/127.5 - 1.
        angle[pos] = convert_xy_to_angle(xy[pos])
        vortex[pos] = calculate_vortex(angle[pos])
        
        angle_vortex_each=np.zeros((L,L,2),dtype=float)
        angle_vortex_each[:,:,0]=angle[pos]/(2*np.pi)
        angle_vortex_each[:,:,1]=vortex[pos]
        angle_vortex[pos]=angle_vortex_each
        
        label[pos, f[key]['label'][()]] = 1
        temperature[pos] = f[key]['T'][()]
        group_mapping.append(key)
        pos += 1
    return xy, angle, vortex, label, temperature, group_mapping,angle_vortex

def load_all_data(list_data_file):
    angle = None
    vortex = None
    label = None
    temperature = None
    angle_vortex=None
    for data_file in list_data_file:
        out = load_data(data_file)
        if angle is None:
            xy=out[0]
            angle = out[1]
            vortex = out[2]
            label = out[3]
            temperature = out[4]
            angle_vortex=out[6]
        else:
            angle = np.vstack((angle, out[1]))
            xy = np.vstack((xy, out[0]))
            vortex = np.vstack((vortex, out[2]))
            label = np.vstack((label, out[3]))
            temperature = np.r_[temperature, out[4]]
            angle_vortex = np.vstack((angle_vortex, out[6]))
    return angle, vortex, label, temperature,angle_vortex,xy


if __name__ == '__main__':
    pass
