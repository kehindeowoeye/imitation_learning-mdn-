from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import objectives
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

import pandas as pd
import keras
from keras.layers import TimeDistributed
from keras.layers import LSTM,RepeatVector
from keras.models import load_model
import json
from keras.models import model_from_json, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras import backend as K
from keras.layers import Merge, Dense
from keras.layers import Reshape
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical
import random
import math
import xlrd
import xlsxwriter
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
import functools
from keras.layers import Bidirectional
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from itertools import product



#######################################################################################
#######################################################################################


#dataset = np.array(pd.read_csv('new1all2s.csv',header=None))
dataset = np.array(pd.read_csv('train22.csv',header=None))
dataset_move = dataset[dataset[:,72] == 2]
dataset_stay = dataset[dataset[:,72] == 1]
#d1 = dataset[0:dataset.shape[0],:]
#d4 = dataset[0:dataset.shape[0],:]
d1 = dataset_move
d4 = dataset_stay
#########################################################################################
#########################################################################################
#Prediction for collective behaviour


look_back = 60
dim_output = 3
n_batch = 1
look_back_sca = 5
num_features = 36
#model = load_model('weights-improvement1-07-0.82.hdf5')



dataset_test = np.array(pd.read_excel('alldata0404.xls',header=None))
#dataset_test = dataset_test[597:dataset_test.shape[0],:]#removes the passage along the corridor
dataset_test = dataset_test[11000:dataset_test.shape[0],:]#starts from when the sheep moved at high velocity
for x in range(0, 1):
    
    dataset0401x =  dataset_test[:,0:36]
    #print(dataset0401x.shape)
    dataset0401y =  dataset_test[:,36:72]
    #labels = np.array(dataset_test[:,72])
    #print(labels.shape)
    rx = np.mean(dataset0401x,axis=1); ry = np.mean(dataset0401y,axis=1);
    rx = np.array([rx])
    rx = rx.reshape(rx.shape[1],1)
    ry = np.array([ry])
    ry = ry.reshape(ry.shape[1],1)
    rx = np.array(dataset0401x)-np.array(rx)
    dd = dataset_test.shape
    trainl = math.trunc(1.0*dd[0])
    #test = dd[0]-trainl
        
    ry = np.array(dataset0401y)-np.array(ry)
    dist = np.sqrt(np.power(rx,2) + np.power(ry,2))
    dist = np.sort(np.array(dist), axis=1)
    Xtrain = np.array(dist[0:trainl,:]);#Xtest = np.array(dist[trainl:,:]);
    #labeltrain = np.array(labels[0:trainl],object);
    #labeltest = np.array(labels[trainl:]);


nb_samples = Xtrain.shape[0] - look_back
x_train_reshaped1 = np.zeros((nb_samples,look_back,num_features))


for i in range(nb_samples):
    y_position = i + look_back
    x_train_reshaped1[i] = Xtrain[i:y_position]
    



num_features = 34*4
dim_output = 2
look_back = 60
#n_epoch = 5000
n_batch = 1
num_classes = 2
n_neurons = 35
numComponents = 10



############################################################################################
############################################################################################
#Generate samples from mixtures

def generate(output, testSize, numComponents=numComponents, outputDim=2, M=1):
    out_pi = output[:,:numComponents]

    out_sigma = output[:,numComponents:2*numComponents]
  
    out_mu = output[:,2*numComponents:]
    out_mu = np.reshape(out_mu, [-1, numComponents, outputDim])
    out_mu = np.transpose(out_mu, [1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = np.amax(out_pi, 1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = np.exp(out_pi)
    normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
    out_pi = normalize_pi * out_pi
    
    # use exponential to make sure sigma is positive
    out_sigma = np.exp(out_sigma)
    result = np.random.rand(testSize, M, outputDim)
    rn = np.random.randn(testSize, M)
    
    mu = 0
    std = 0
    idx = 0
    for j in range(0, M):
        for i in range(0, testSize):
            for d in range(0, outputDim):
                print(out_pi)
                print(out_mu)
               
                #print(out_sigma)
                
                idx = np.random.choice(numComponents, 1, p=out_pi[i])
                print(idx)
                mu = out_mu[idx,i,d]
                std = out_sigma[i, idx]
                #print(mu)
                #print(std)
               
                result[i, j, d] = mu + rn[i, j]*std
    return result

############################################################################################



def generate1(output, testSize, numComponents=numComponents, outputDim=2, M=1):
    out_pi = output[:,:,0:numComponents]
    out_pi = out_pi[-1]
   
    out_sigma = output[:,:,numComponents:2*numComponents]
   
    out_sigma = out_sigma[-1]
    out_mu = output[:,:,2*numComponents:]
 
    out_mu = out_mu[-1]
    
    out_mu = np.reshape(out_mu, [-1, numComponents, outputDim])
    out_mu = np.transpose(out_mu, [1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = np.amax(out_pi, 1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = np.exp(out_pi)
    normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
    out_pi = normalize_pi * out_pi
    
    # use exponential to make sure sigma is positive
    out_sigma = np.exp(out_sigma)
    result = np.random.rand(testSize, M, outputDim)
    rn = np.random.randn(testSize, M)
    
    mu = 0
    std = 0
    idx = 0
    for j in range(0, M):
        for i in range(0, testSize):
            for d in range(0, outputDim):
                
                
                idx = np.random.choice(numComponents, 1, p=out_pi[i])
                print(idx)
                
                
                
                mu = out_mu[idx,i,d]
                std = out_sigma[i, idx]
                result[i, j, d] = mu + rn[i, j]*std
    return result
############################################################################################

dd = dataset.shape
trainl = math.trunc(1.0*dd[0])

############################################################################################

m1    = load_model('weights-v1.hdf5')
m10   = load_model('weights-v10.hdf5')
m101  = load_model('weights-v101.hdf5')
m1001 = load_model('weights-v1001.hdf5')
m1000 = load_model('weights-v1000.hdf5')
m100  = load_model('weights-improvement1-12-0.76.hdf5')
model = load_model('model_high_acc.hdf5')
#model_scat  = load_model('weights-improvement1-11-0.71.hdf5')
#model_scat = load_model('weights-improvement1-128-0.71.hdf5')
model_scat = load_model('weights-improvement1-22-0.63.hdf5')#individual
#model_scat = load_model('weights-improvement1-12-0.71.hdf5')
#model_scat = load_model('weights-improvement1-25-0.61.hdf5')
#model_scat = load_model('weights-improvement1-25-0.79.hdf5')




from mdn1 import *
model_move = Sequential()
model_move.add(Dense(256,input_shape=(num_features,)))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dropout(0.2))
model_move.summary()
model_move.add(MixtureDensity(dim_output,4))
model_move.load_weights('m1')
gmm1 = model_move


from mdn2 import *
model_move = Sequential()
model_move.add(Dense(256,input_shape=(num_features,)))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dropout(0.2))
model_move.summary()
model_move.add(MixtureDensity(dim_output,8))
model_move.load_weights('m228')
gmm2 = model_move



from mdn3 import *
model_move = Sequential()
model_move.add(Dense(256,input_shape=(num_features,)))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dropout(0.2))
model_move.summary()
model_move.add(MixtureDensity(dim_output,1))
model_move.load_weights('m31')
gmm3 = model_move

from mdn4 import *
model_move = Sequential()
model_move.add(Dense(256,input_shape=(num_features,)))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dropout(0.2))
model_move.summary()
model_move.add(MixtureDensity(dim_output,6))
model_move.load_weights('m4')
gmm4 = model_move


from mdn5 import *
model_move = Sequential()
model_move.add(Dense(256,input_shape=(num_features,)))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dropout(0.2))
model_move.summary()
model_move.add(MixtureDensity(dim_output,5))
model_move.load_weights('m5')
gmm5 = model_move


from mdn6 import *
model_move = Sequential()
model_move.add(Dense(256,input_shape=(num_features,)))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dropout(0.2))
model_move.summary()
model_move.add(MixtureDensity(dim_output,5))
model_move.load_weights('m6')
gmm6 = model_move



from mdn11 import *
model_move = Sequential()
model_move.add(Dense(256,input_shape=(num_features,)))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dense(256))
model_move.add(Activation('relu'))
model_move.add(Dropout(0.2))
model_move.summary()
model_move.add(MixtureDensity(dim_output,1))
#model_move.load_weights('weights-improvement2-04--4.13.hdf5')
model_move.load_weights('weights-improvement2-17--4.25.hdf5')
gmm11 = model_move
############################################################################


kb = np.zeros((dataset_test.shape[0],2))

y_test = np.array([0,0])
num_features = 34*4

dataset0401x =  dataset_test[:,0:36]
#print(dataset0401x.shape)
dataset0401y =  dataset_test[:,36:72]
w = 0;x = 0;y_position = w + look_back
rx1 = dataset0401x[:,x];ry1 = dataset0401y[:,x];
#print(rx1.shape)
rx1 = rx1.reshape(rx1.shape[0],1)
ry1 = ry1.reshape(ry1.shape[0],1)

y_position = w + look_back
rx1 = rx1[w:y_position,:];ry1 = ry1[w:y_position,:]
#print(rx1.shape);
bl = np.array([0,0]);
bl[0] = rx1[rx1.shape[0]-1];bl[1] = ry1[ry1.shape[0]-1]
bl = bl.reshape(1,2)
print('this si bl',bl)
#print(rx1)

#nb_samplesnn = dataset_test.shape[0] - look_back
x_train_reshapednn = np.zeros((1,look_back,num_features))
#for w in range(0,dataset_test.shape[0]-look_back):     note that i decided to shorten length here see next line
for w in range(0,dataset_test.shape[0]-look_back-40000):
    print("test data seen so far", w)
    dataset0401x =  dataset_test[:,0:36];dataset0401x =np.delete( dataset0401x,13,axis=1)
    dataset0401y =  dataset_test[:,36:72];dataset0401y =np.delete(dataset0401y,13,axis=1)
    rx1 = np.delete(rx1,0);ry1 = np.delete(ry1,0)
    #print(rx1.shape)
    
    rx1 = rx1.reshape(rx1.shape[0],1)
    ry1 = ry1.reshape(ry1.shape[0],1)
    
    #print(rx1.shape)
    if w>0:
        rx1 = np.vstack((rx1,kb[w-1,0]));ry1 = np.vstack((ry1,kb[w-1,1]));
    else:
        rx1 = np.vstack((rx1,bl[0,0]));ry1 = np.vstack((ry1,bl[0,1]));
    #print(rx1,ry1)
    y_position = w + look_back
    dataset0401x =  np.delete(dataset0401x,(x),axis = 1);
    dataset0401y =  np.delete(dataset0401y,(x),axis = 1);
    acx = dataset0401x[w:y_position+1,:];acy = dataset0401y[w:y_position+1,:]
    acx = np.diff(acx,axis = 0); acy = np.diff(acy,axis = 0)
    dataset0401x =  dataset0401x[w:y_position,:];dataset0401y = dataset0401y[w:y_position,:];
   
    rx1 = np.array([rx1])
    rx1 = rx1.reshape(rx1.shape[1],1)
    ry1 = np.array([ry1])
    ry1 = ry1.reshape(ry1.shape[1],1)
    
    rx11 = np.array(dataset0401x)-np.array(rx1)
    ry11 = np.array(dataset0401y)-np.array(ry1)
    
    ag = np.arctan2(ry,rx)
    #rx11 = np.sort(rx11[0:trainl,:],axis = 1);ry11 = np.sort(ry11[0:trainl,:],axis = 1)
  
    #Xtrainnn = np.concatenate((rx11, ry11),axis = 1);
    distnnn = np.sqrt(np.power(rx11,2)+ np.power(ry11,2))
    b1 = np.argsort(distnnn);
   
    acx1 = np.zeros((acx.shape[0],acx.shape[1]))
    acy1 = np.zeros((acy.shape[0],acy.shape[1]))
    ag1 = np.zeros((acy.shape[0],acy.shape[1]))
 

    for i in range(0,acx.shape[0]):
        bdx = acx[i,:];bdy = acy[i,:];agb = ag[i,:]
        acx1[i,:] = bdx[b1[i,:]]
        acy1[i,:] = bdy[b1[i,:]]
        ag1[i,:]  = agb[b1[i,:]]
    
    acx = acx1;acy = acy1;ag = ag1;
    distnnn = np.sort(np.array(distnnn), axis=1)
    Xtrainnn = np.array(distnnn[0:trainl,:]);
    
    Xtrainnn = np.concatenate((Xtrainnn,acx,acy,ag),axis = 1)
    print(Xtrainnn.shape)
    #print(Xtrainnn.shape)
    #nb_samplesnn = Xtrain.shape[0] - look_back
    
    
    #for i in range(nb_samplesnn):
    x_train_reshapednn = np.zeros((1,look_back,num_features))
    #print(y_position)


    x_train_reshapednn[0] = Xtrainnn[0:y_position]
    #x_train_reshapednnn = x_train_reshapednn[0]
    #y_test = np.array([0,0])









##########################################################################################


    #for g in range(0,x_train_reshaped1.shape[0]):
    for g in range(0,1):
        capo = np.zeros((1,look_back,36))
        capo[0] = x_train_reshaped1[w,:]
        #print(capo.shape)
        bv = model.predict(capo,batch_size=1)#first layer of prediction
        bv = bv[0,:]
        bv = np.argmax(bv[look_back-1,:])
        print("The collective behaviour predicted", bv)
        if bv == 0 or bv == 1:
            #print(x_train_reshapednn.shape)
            ca0 = model_scat.predict(x_train_reshapednn[:,-5:,:])#mid(second)layer of prediction
            ca0 = ca0[0,:]
            ca0 = np.argmax(ca0[look_back_sca-1,:])
            if ca0 == 0:
                print("not moving when scattered")
                print(bl)
                if w == 0:
                    kb[w,0] = bl[w,0] +  0.0
                    kb[w,1] = bl[w,1] +  0.0
                else:
                    kb[w,0] = kb[w-1,0] +  0.0
                    kb[w,1] = kb[w-1,1] +  0.0
                print(kb[w])
           
       
                
            else:
                print("moving when scattered")
                ca0 = m1.predict(x_train_reshapednn[:,-look_back_sca:,:])#mid(second)layer of prediction
                ca0 = ca0[0,:]
                ca0 = np.argmax(ca0[look_back_sca-1,:])
                if ca0==1:
                    cd = x_train_reshapednn[:,-1:,:]
                    cd = cd[0,:]
                    #cd = x_train_reshapednn[:,-5:,:]


                    y_test = generate(gmm11.predict(cd,batch_size=1),
                                  1,
                                  numComponents=1,
                                  outputDim=dim_output)
                    y_test = np.reshape(y_test,(1,2))
                    print('this is m11')
                    if w == 0:
                        kb[w,0] = bl[w,0] + y_test[0,0]
                        kb[w,1] = bl[w,1] + y_test[0,1]
                    else:
                        print(kb[w-1,0])
                        kb[w,0] = kb[w-1,0] + y_test[0,0] 
                        kb[w,1] = kb[w-1,1] + y_test[0,1]
                    print(kb[w])
                  
                else:
                    ca0 = m10.predict(x_train_reshapednn[:,-look_back_sca:,:])#mid(second)layer of prediction
                    ca0 = ca0[0,:]
                    ca0 = np.argmax(ca0[look_back_sca-1,:])
                    if ca0==0:
                        ca0 = m100.predict(x_train_reshapednn[:,-look_back_sca:,:])
                        ca0 = ca0[0,:]
                        ca0 = np.argmax(ca0[look_back_sca-1,:])
                        if ca0==0:
                            ca0 = m1000.predict(x_train_reshapednn[:,-look_back_sca:,:])
                            ca0 = ca0[0,:]
                            ca0 = np.argmax(ca0[look_back_sca-1,:])
                            if ca0==0:
                                cd = x_train_reshapednn[:,-1:,:];cd = cd[0,:]
                                y_test = generate(gmm3.predict(cd,batch_size=1),
                                    1,
                                numComponents=1,
                                outputDim=dim_output)
                                print('this is m10000')
                                y_test = np.reshape(y_test,(1,2))
                                if w == 0:
                                    kb[w,0] = bl[w,0] + y_test[0,0]
                                    kb[w,1] = bl[w,1] + y_test[0,1]
                                else:
                                    print(kb[w-1,0])
                                    kb[w,0] = kb[w-1,0] + y_test[0,0]
                                    kb[w,1] = kb[w-1,1] + y_test[0,1]
                                print(kb[w])
                        
                            else:
                                cd = x_train_reshapednn[:,-1:,:];cd = cd[0,:]
                                y_test = generate(gmm4.predict(cd,batch_size=1),
                                                      1,
                                                      numComponents=6,
                                                      outputDim=dim_output)
                                print('this is m10001')
                                y_test = np.reshape(y_test,(1,2))
                                if w == 0:
                                        kb[w,0] = bl[w,0] + y_test[0,0]
                                        kb[w,1] = bl[w,1] + y_test[0,1]
                                else:
                                        print(kb[w-1,0])
                                        kb[w,0] = kb[w-1,0] + y_test[0,0]
                                        kb[w,1] = kb[w-1,1] + y_test[0,1]
                                print(kb[w])
                        else:
                             ca0 = m1001.predict(x_train_reshapednn[:,-look_back_sca:,:])
                             ca0 = ca0[0,:]
                             ca0 = np.argmax(ca0[look_back_sca-1,:])
                             if ca0==0:
                                 cd = x_train_reshapednn[:,-1:,:];cd = cd[0,:]
                                 y_test = generate(gmm5.predict(cd,batch_size=1),
                                         1,
                                        numComponents=5,
                                        outputDim=dim_output)
                                 print('this is m10010')
                                 y_test = np.reshape(y_test,(1,2))
                                 if w == 0:
                                    kb[w,0] = bl[w,0] + y_test[0,0]
                                    kb[w,1] = bl[w,1] + y_test[0,1]
                                 else:
                                    print(kb[w-1,0])
                                    kb[w,0] = kb[w-1,0] + y_test[0,0]
                                    kb[w,1] = kb[w-1,1] + y_test[0,1]
                                 print(kb[w])
                             else:
                                cd = x_train_reshapednn[:,-1:,:];cd = cd[0,:]
                                y_test = generate(gmm6.predict(cd,batch_size=1),
                                    1,
                                numComponents=5,
                                outputDim=dim_output)
                                print('this is m10011')
                                y_test = np.reshape(y_test,(1,2))
                                if w == 0:
                                    kb[w,0] = bl[w,0] + y_test[0,0]
                                    kb[w,1] = bl[w,1] + y_test[0,1]
                                else:
                                    print(kb[w-1,0])
                                    kb[w,0] = kb[w-1,0] + y_test[0,0]
                                    kb[w,1] = kb[w-1,1] + y_test[0,1]
                                print(kb[w])
                            








                    
                    else:
                        ca0 = m101.predict(x_train_reshapednn[:,-look_back_sca:,:])
                        ca0 = ca0[0,:]
                        ca0 = np.argmax(ca0[look_back_sca-1,:])
                        if ca0==0:
                                cd = x_train_reshapednn[:,-1:,:];cd = cd[0,:]
                                y_test = generate(gmm1.predict(cd,batch_size=1),
                                                      1,
                                                      numComponents=4,
                                                      outputDim=dim_output)
                                y_test = np.reshape(y_test,(1,2))
                                print('this is m1010')
                                if w == 0:
                                    kb[w,0] = bl[w,0] + y_test[0,0]
                                    kb[w,1] = bl[w,1] + y_test[0,1]
                                else:
                                    print(kb[w-1,0])
                                    kb[w,0] = kb[w-1,0] + y_test[0,0]
                                    kb[w,1] = kb[w-1,1] + y_test[0,1]
                                print(kb[w])
                        else:
                            cd = x_train_reshapednn[:,-1:,:];cd = cd[0,:]
                            y_test = generate(gmm2.predict(cd,batch_size=1),
                                                      1,
                                                      numComponents=8,
                                                      outputDim=dim_output)
                            y_test = np.reshape(y_test,(1,2))
                        
                            print('this is m1011')
                            print(y_test)
                           
                            if w == 0:
                                    kb[w,0] = bl[w,0]
                                    kb[w,1] = bl[w,1]
                        
                            
                            else:
                                    print(kb[w-1,0])
                                    kb[w,0] = kb[w-1,0] + y_test[0,0]
                                    kb[w,1] = kb[w-1,1] + y_test[0,1]
                            print(kb[w])
                                    
                        
    
        else:
            if w == 0:
                kb[w,0] = bl[w,0] + y_test[0]
                kb[w,1] = bl[w,1] + y_test[1]
            else:
                kb[w,0] = kb[w-1,0] + 0.0
                kb[w,1] = kb[w-1,1] + 0.0
            print(kb[w])








import xlsxwriter
workbook = xlsxwriter.Workbook('thu_afta.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(kb.T):
    worksheet.write_column(row, col, data)
workbook.close()





