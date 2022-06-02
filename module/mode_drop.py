############################################################
################# Mode Drop Data Generator #################
############################################################
def mode_drop_gen(data, method = 'sequential', ratio = 1, isnumpy = False):
    # if method = 'squential', sequential mode drop
    # if method = 'simultaneous', simultaneous mode drop
    # When 'sequential' choice of ratio need to be integer with upperbound of max(number of classes)
    # When 'simultaneous' choice of ratio need to be real number with range of [0,1]
    import numpy as np
    import torch
    
    # data as numpy array
    if (isnumpy == False):
        if (isinstance(data, list) == True):
            data = np.asarray(data)
        elif (isinstance(data, tuple) == True):
            data = np.asarray(data)
        elif (torch.is_tensor(data) == True):
            for batch_idx, Input in enumerate(data):
                if (batch_idx == 0):
                    convert_data = Input.detach().numpy()
                else:
                    convert_data = np.vstack((convert_data, Input.detach().numpy()))
            data = convert_data
    elif (isnumpy == True): pass

    # assign class
    Class = np.zeros(len(data))
    for iloop in range(len(data)):
        if (data[iloop,0]<2 and data[iloop,1]<2):
            Class[iloop] = 1 
        elif (data[iloop,0]<2 and data[iloop,1]>8):
            Class[iloop] = 2
        elif (data[iloop,0]>8.2 and data[iloop,1]<2):
            Class[iloop] = 3 
        elif (data[iloop,0]>8.2 and data[iloop,1]>8):
            Class[iloop] = 4
        else: pass
    circle0 = np.array([])
    circle1 = np.array([])
    circle2 = np.array([])
    circle3 = np.array([])
    circle4 = np.array([])
    for iloop in range(len(Class)):
        # Middle circle
        if (Class[iloop] == 0):
            circle0 = np.append(circle0, iloop)
        elif (Class[iloop] == 1):
            circle1 = np.append(circle1, iloop)
        elif (Class[iloop] == 2):
            circle2 = np.append(circle2, iloop)
        elif (Class[iloop] == 3):
            circle3 = np.append(circle3, iloop)
        elif (Class[iloop] == 4):
            circle4 = np.append(circle4, iloop)
    circle0 = list(map(int, circle0))
    circle1 = list(map(int, circle1))
    circle2 = list(map(int, circle2))
    circle3 = list(map(int, circle3))
    circle4 = list(map(int, circle4))

    # sequential Drop
    if (method == 'sequential'):
        if (ratio == 0):
            return data
        elif (ratio == 1):
            temp_data = data[circle0,:]
            temp_data = np.vstack((temp_data, data[circle1,:]))
            temp_data = np.vstack((temp_data, data[circle2,:]))
            temp_data = np.vstack((temp_data, data[circle3,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle4), replace = False),:] 
            temp[:,0] = temp[:,0] + np.random.normal(loc = 0, scale = 0.1, size = len(circle4))
            temp[:,1] = temp[:,1] + np.random.normal(loc = 0, scale = 0.1, size = len(circle4))
            temp_data = np.vstack((temp_data, temp))
            return temp_data
        elif (ratio == 2):
            temp_data = data[circle0,:]
            temp_data = np.vstack((temp_data, data[circle1,:]))
            temp_data = np.vstack((temp_data, data[circle2,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle3)+len(circle4), replace = False),:] 
            temp[:,0] = temp[:,0] + np.random.normal(loc = 0, scale = 0.1, size = len(circle3)+len(circle4))
            temp[:,1] = temp[:,1] + np.random.normal(loc = 0, scale = 0.1, size = len(circle3)+len(circle4))
            temp_data = np.vstack((temp_data, temp))
            return temp_data
        elif (ratio == 3):
            temp_data = data[circle0,:]
            temp_data = np.vstack((temp_data, data[circle1,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle2)+len(circle3)+len(circle4), replace = False),:] 
            temp[:,0] = temp[:,0] + np.random.normal(loc = 0, scale = 0.1, size = len(circle2)+len(circle3)+len(circle4))
            temp[:,1] = temp[:,1] + np.random.normal(loc = 0, scale = 0.1, size = len(circle2)+len(circle3)+len(circle4))
            temp_data = np.vstack((temp_data, temp))
            return temp_data
        elif (ratio == 4):
            temp_data = data[circle0,:]
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle1)+len(circle2)+len(circle3)+len(circle4), replace = False),:]
            temp[:,0] = temp[:,0] + np.random.normal(loc = 0, scale = 0.1, size = len(circle1)+len(circle2)+len(circle3)+len(circle4))
            temp[:,1] = temp[:,1] + np.random.normal(loc = 0, scale = 0.1, size = len(circle1)+len(circle2)+len(circle3)+len(circle4))
            temp_data = np.vstack((temp_data, temp))
            return temp_data

    # simultaneous drop        
    if (method == 'simultaneous'):
        temp_data = data[circle0,:]
        sample_data = data[circle1,:]
        sample_data = np.vstack((sample_data, data[circle2,:]))
        sample_data = np.vstack((sample_data, data[circle3,:]))
        sample_data = np.vstack((sample_data, data[circle4,:]))
        temp = sample_data[np.random.choice(range(len(sample_data)), int(round(len(sample_data)*(1-ratio))), replace = False),:]
        temp2 = temp_data[np.random.choice(range(len(data[circle0,:])), round(len(sample_data)*ratio), replace = True),:]
        temp2[:,0] = temp2[:,0] + np.random.normal(loc = 0, scale = 0.1, size = round(len(sample_data)*ratio))
        temp2[:,1] = temp2[:,1] + np.random.normal(loc = 0, scale = 0.1, size = round(len(sample_data)*ratio)) 
        temp_data = np.vstack((temp_data, temp))
        temp_data = np.vstack((temp_data, temp2))
        return temp_data