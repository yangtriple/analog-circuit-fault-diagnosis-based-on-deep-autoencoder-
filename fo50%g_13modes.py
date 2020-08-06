# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:06:45 2020

@author: yangt
"""
import numpy as np
from keras.utils.np_utils import to_categorical
from myfunction import load_data,dataselected,showpic,TBAE
np.random.seed(1337)  # for reproducibility
from keras.layers import Input,Dense    
from keras.models import Model   
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt


'''
fouropamp 滤波器实验

'''
folder_path=r'E:\analogdata\fouropamp1500\all'
filename=r'fo50.csv'
df_50=load_data(folder_path,filename)
print('数据加载大小:',df_50.shape)

x_train,x_test,y_train,y_test=dataselected(df_50,n_cut=0,n_down=7,n_s=200,n_f=200,n_c=13)

print('训练集数据大小：',x_train.shape)
print('测试集数据大小:',x_test.shape)
y_train_sc=to_categorical(y_train,num_classes=13)#把类别标签转换为onehot编码，onehot编码是一种计算机处理的二元编码
y_test_sc=to_categorical(y_test,num_classes=13)


''''
自编码器模型
'''
result=TBAE(x_train,x_test,y_train_sc,y_test_sc,class_n=13,hidden_layer=[100,50],epoc=200,batch_s=100, 
         lr_a=0.005,decay_a=1e-6)
score=result[3]
print('准确率',score)
