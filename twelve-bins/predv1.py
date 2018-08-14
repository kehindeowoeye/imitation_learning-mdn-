from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import objectives
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from mdn import *
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
import dill as pickle
from IPython.display import clear_output
######################################################################################

# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=False)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

plot_losses = PlotLosses()




#######################################################################################
#######################################################################################

kb = 0

dataset = np.array(pd.read_excel('vv1.xlsx',header=None))
bigd = np.array(pd.read_csv('alldata0401.csv',header=None))
bigdx = np.diff(bigd,axis = 0)



x1 = 0
dim_output = 2
look_back = 5
n_epoch = 20
n_batch = 10
num_classes = 2
n_neurons = 35
num_features = 34*4

#########################################################################################
#########################################################################################
#Prediction for collective behaviour


tu = np.arange(1,37)
tu = np.delete(tu,13)


for y in tu:
    d1 = dataset[dataset[:,1]==y]
    dd = d1.shape
    trainl = math.trunc(1.0*dd[0])
    ds = d1[:,0]
    dt = d1[:,4]
   
    
    for yy in np.array(ds):
      
        if int(yy) == int(ds[0]):
            db = bigd[int(yy),:].reshape(1,len(bigd[int(yy),:]))
            dbx = bigdx[int(yy),:].reshape(1,len(bigd[int(yy),:]))
        else:
            cm = bigd[int(yy),:]
            db = np.concatenate((db,cm.reshape(1,73)), axis=0)
            cmx = bigdx[int(yy),:]
            dbx = np.concatenate((dbx,cmx.reshape(1,73)), axis=0)

       





    for x in range(0,1):
        
        dataset0401x =  db[:,0:36];dataset0401x =np.delete( dataset0401x,13,axis=1)
        dataset0401y =  db[:,36:72];dataset0401y =np.delete(dataset0401y,13,axis=1)
        
        dax =  dbx[:,0:36];dax =np.delete(dax,13,axis=1)
        day =  dbx[:,36:72];day =np.delete(day,13,axis=1)
        
        #gx = np.array(dataset0401x[:,x1]);
        #gy = np.array(dataset0401y[:,x1])
        #gx = np.array(gx);gy = np.array(gy);gx = gx.reshape(len(gx),1);gy = gy.reshape(len(gy),1)
        labels = dt.reshape(len(dt),1);bn=[];
        rx = dataset0401x[:,x1]; dataset0401x = np.delete(dataset0401x,(x1),axis = 1);
        ry = dataset0401y[:,x1]; dataset0401y = np.delete(dataset0401y,(x1),axis = 1);
        acx = np.delete(dax,(x1),axis = 1);acy = np.delete(day,(x1),axis = 1);
    
    
        rx = np.array([rx])
        rx = rx.reshape(rx.shape[1],1)
        ry = np.array([ry])
        ry = ry.reshape(ry.shape[1],1)
        rx = np.array(dataset0401x)-np.array(rx)
    
        ry = np.array(dataset0401y)-np.array(ry)
        dist = np.sqrt(np.power(rx,2)+ np.power(ry,2))
        ag = np.arctan2(ry,rx)
    
        b1 = np.argsort(dist);
        acx1 = np.zeros((acx.shape[0],acx.shape[1]))
        acy1 = np.zeros((acy.shape[0],acy.shape[1]))
        ag1 = np.zeros((acy.shape[0],acy.shape[1]))
    
    
        for i in range(0,acx.shape[0]):
            bdx = acx[i,:];bdy = acy[i,:];agb = ag[i,:]
            acx1[i,:] = bdx[b1[i,:]]
            acy1[i,:] = bdy[b1[i,:]]
            ag1[i,:]  = agb[b1[i,:]]

        acx = acx1;acy = acy1;ag = ag1;

        dist = np.sort(np.array(dist), axis=1)
        Xtrain1 = np.array(dist);Xtest = np.array(dist[trainl:,:]);
        Xtrain1 = np.concatenate((Xtrain1,acx,acy,ag),axis= 1)
        labeltrain1 = np.array(labels[0:trainl],object);

        if x == 0:
        
            alldata = Xtrain1
            alllabels = labeltrain1
        else:
            alldata = np.concatenate((alldata,Xtrain1),axis=0);Xtrain1=[];
            alllabels = np.concatenate((alllabels,labeltrain1),axis=0);labeltrain1=[];

        if int(y) == int(tu[0]):
            alldata1 = alldata
            alllabels1 = alllabels
        else:
            alldata1 = np.concatenate((alldata1,alldata),axis=0);
            alllabels1 = np.concatenate((alllabels1,alllabels),axis=0);
            print(alllabels1.shape)
        x1 = x1+1


Xtrain = alldata1
labeltrain = alllabels1
print(labeltrain.shape)





nb_samples = Xtrain.shape[0] - look_back
x_train_reshaped2 = np.zeros((nb_samples,look_back,num_features))
y_train_reshaped2 = np.zeros((nb_samples,1,dim_output))
one_hot_labels2 = np.zeros((nb_samples,1,num_classes))
one_hot_label = (pd.get_dummies(np.array(labeltrain.astype(int).reshape(-1))))


for i in range(nb_samples):
    y_position = i + look_back
    x_train_reshaped2[i] = Xtrain[i:y_position]
    one_hot_labels2[i] = one_hot_label.iloc[y_position,:2]



model = Sequential()
opt = Adam(lr=0.001)

model.add(Bidirectional(LSTM(4,return_sequences=True,activation = 'tanh'), input_shape=(None, num_features)))



model.add(TimeDistributed(Dense(num_classes,activation = 'tanh')))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath="weights-improvement1-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(x_train_reshaped2, one_hot_labels2, epochs=n_epoch, batch_size=n_batch, verbose=2,validation_split = 0.1,callbacks=callbacks_list)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

model.save('c1')

