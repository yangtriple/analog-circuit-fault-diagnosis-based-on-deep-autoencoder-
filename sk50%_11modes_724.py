# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:19:45 2020

@author: yangt
"""

import numpy as np
from keras.utils.np_utils import to_categorical
from myfunction import load_data,dataselected,TBAE,showpic
np.random.seed(1337)  # for reproducibility

'''
sallen-key 滤波器实验

'''
folder_path=r'E:\analogdata\AC_t5%\sallen-key\all600'
filename=r'sk50%_11modes723.csv'
df_50=load_data(folder_path,filename)
print('load data size:',df_50.shape)

x_train,x_test,y_train,y_test=dataselected(df_50,n_cut=17,n_down=7,n_s=150,n_f=85,n_c=11)

print('training set size：',x_train.shape)
print('testing set size:',x_test.shape)
y_train_sc=to_categorical(y_train,num_classes=11)#把类别标签转换为onehot编码，onehot编码是一种计算机处理的二元编码
y_test_sc=to_categorical(y_test,num_classes=11)
print('training label size：',y_train_sc.shape)
print('testing label size:',y_test_sc.shape)

'''
add noise to input signals
'''
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 


''''
自编码器模型
'''
mymodel=TBAE(x_train_noisy,x_test,y_train_sc,y_test_sc,class_n=11,hidden_layer=[50,30],epoc=200,batch_s=100, 
         lr_a=0.001,decay_a=1e-6)


