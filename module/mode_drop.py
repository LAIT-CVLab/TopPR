import numpy as np
import torch
import pickle
from torchvision import datasets, transforms

############################################################
############ Gaussian Mode Drop Data Generator #############
############################################################
def gaussian_mode_drop(method, ratio = 1, isnumpy = False):
    # generate mixture of gaussian distribution with multi modes (7 modes)
    num_samples = 1000
    feature_dim = 64
    features = np.random.normal(loc=0.0, scale=1.0, size=[num_samples, feature_dim])
    features = np.vstack((features, np.random.normal(loc=30.0, scale=1.0, size=[num_samples, feature_dim])))
    features = np.vstack((features, np.random.normal(loc=20.0, scale=1.0, size=[num_samples, feature_dim])))
    features = np.vstack((features, np.random.normal(loc=10.0, scale=1.0, size=[num_samples, feature_dim])))
    features = np.vstack((features, np.random.normal(loc=-10.0, scale=1.0, size=[num_samples, feature_dim])))
    features = np.vstack((features, np.random.normal(loc=-20.0, scale=1.0, size=[num_samples, feature_dim])))
    features = np.vstack((features, np.random.normal(loc=-30.0, scale=1.0, size=[num_samples, feature_dim])))

    # assign class
    Class = np.zeros(len(features))
    Class[0:1000] = 0
    Class[1000:1000*2] = 1
    Class[1000*2:1000*3] = 2
    Class[1000*3:1000*4] = 3
    Class[1000*4:1000*5] = 4
    Class[1000*5:1000*6] = 5
    Class[1000*6:1000*7] = 6

    circle0 = np.array([])
    circle1 = np.array([])
    circle2 = np.array([])
    circle3 = np.array([])
    circle4 = np.array([])
    circle5 = np.array([]) 
    circle6 = np.array([])

    for iloop in range(len(Class)):
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
        elif (Class[iloop] == 5):
            circle5 = np.append(circle5, iloop)
        elif (Class[iloop] == 6):
            circle6 = np.append(circle6, iloop)
        
    circle0 = list(map(int, circle0))
    circle1 = list(map(int, circle1))
    circle2 = list(map(int, circle2))
    circle3 = list(map(int, circle3))
    circle4 = list(map(int, circle4))
    circle5 = list(map(int, circle5))
    circle6 = list(map(int, circle6))

    # sequential Drop
    if (method == 'sequential'):
        if (ratio == 0):
            return features
        elif (ratio == 1):
            temp_data = features[circle0,:]
            temp_data = np.vstack((temp_data, features[circle1,:]))
            temp_data = np.vstack((temp_data, features[circle2,:]))
            temp_data = np.vstack((temp_data, features[circle3,:]))
            temp_data = np.vstack((temp_data, features[circle4,:]))
            temp_data = np.vstack((temp_data, features[circle5,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle6), replace = True),:]
            temp = temp + np.random.normal(loc = 0, scale = 0.1, size = [len(circle6), feature_dim])
            temp_data = np.vstack((temp_data, temp))
            return temp_data

        elif (ratio == 2):
            temp_data = features[circle0,:]
            temp_data = np.vstack((temp_data, features[circle1,:]))
            temp_data = np.vstack((temp_data, features[circle2,:]))
            temp_data = np.vstack((temp_data, features[circle3,:]))
            temp_data = np.vstack((temp_data, features[circle4,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle5) + len(circle6), replace = True),:]
            temp = temp + np.random.normal(loc = 0, scale = 0.1, size = [len(circle5) + len(circle6), feature_dim])
            temp_data = np.vstack((temp_data, temp))
            return temp_data

        elif (ratio == 3):
            temp_data = features[circle0,:]
            temp_data = np.vstack((temp_data, features[circle1,:]))
            temp_data = np.vstack((temp_data, features[circle2,:]))
            temp_data = np.vstack((temp_data, features[circle3,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle4) + len(circle5) + len(circle6), replace = True),:]
            temp = temp + np.random.normal(loc = 0, scale = 0.1, size = [len(circle4) + len(circle5) + len(circle6), feature_dim])
            temp_data = np.vstack((temp_data, temp))
            return temp_data

        elif (ratio == 4):
            temp_data = features[circle0,:]
            temp_data = np.vstack((temp_data, features[circle1,:]))
            temp_data = np.vstack((temp_data, features[circle2,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle3) + len(circle4) + len(circle5) + len(circle6), replace = True),:]
            temp = temp + np.random.normal(loc = 0, scale = 0.1, size = [len(circle3) + len(circle4) + len(circle5) + len(circle6), feature_dim])
            temp_data = np.vstack((temp_data, temp))
            return temp_data
        
        elif (ratio == 5):
            temp_data = features[circle0,:]
            temp_data = np.vstack((temp_data, features[circle1,:]))
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle2) + len(circle3) + len(circle4) + len(circle5) + len(circle6), replace = True),:]
            temp = temp + np.random.normal(loc = 0, scale = 0.1, size = [len(circle2) + len(circle3) + len(circle4) + len(circle5) + len(circle6), feature_dim])
            temp_data = np.vstack((temp_data, temp))
            return temp_data

        elif (ratio == 6):
            temp_data = features[circle0,:]
            temp = temp_data[np.random.choice(range(len(temp_data)), len(circle1) + len(circle2) + len(circle3) + len(circle4) + len(circle5) + len(circle6), replace = True),:]
            temp = temp + np.random.normal(loc = 0, scale = 0.1, size = [len(circle1) + len(circle2) + len(circle3) + len(circle4) + len(circle5) + len(circle6), feature_dim])
            temp_data = np.vstack((temp_data, temp))
            return temp_data

    # simultaneous drop        
    if (method == 'simultaneous'):
        temp_data = features[circle0,:]
        sample_data = features[circle1,:]
        sample_data = np.vstack((sample_data, features[circle2,:]))
        sample_data = np.vstack((sample_data, features[circle3,:]))
        sample_data = np.vstack((sample_data, features[circle4,:]))
        sample_data = np.vstack((sample_data, features[circle5,:]))
        sample_data = np.vstack((sample_data, features[circle6,:]))

        temp = sample_data[np.random.choice(range(len(sample_data)), int(round(len(sample_data)*(1-ratio))), replace = False),:]
        temp2 = temp_data[np.random.choice(range(len(features[circle0,:])), round(len(sample_data)*ratio), replace = True),:]
        temp2 = temp2 + np.random.normal(loc = 0, scale = 0.1, size = [round(len(sample_data)*ratio), feature_dim])
        temp_data = np.vstack((temp_data, temp))
        temp_data = np.vstack((temp_data, temp2))
        return temp_data