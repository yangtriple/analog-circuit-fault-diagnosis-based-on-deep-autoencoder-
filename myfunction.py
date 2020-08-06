
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:32:12 2020

@author: yangt
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
np.random.seed(1337)  # for reproducibility
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Input
from keras.models import Model   
from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
import math
import random
import keras
from sklearn.metrics import confusion_matrix,accuracy_score
def load_data(folder_path,filename):
    df_r = pd.read_csv(folder_path + '\\'+ filename,header=None)
    df_n=np.array(df_r)
    data=df_n[:,:-1]
    label=df_n[:,-1]
    print('读取文件大小：',df_n.shape)
    print('读取数据大小：',data.shape)
    print('读取标签大小：',label.shape)
    return df_n
def dataselected(df_n,n_cut=17,n_down=1,n_s=50,n_f=600,n_c=11):
    '''
    data为输入全部数据，label为输入全部标签
    n_down表示降采样数，n_s表示每个故障类型取多少组数据，n_f表示取多少个采样点，n_c表示总共取多少个故障类型,n_train训练测试数据划分百分比
    '''
    #首先，信号降采样
    data_cut=df_n[:,n_cut:-1]
    x=[]
    for i in range(0,data_cut.shape[1]):
        if i%n_down==0:
            x.append(data_cut[:,i])
    data_down=np.array(x).T
    data_down=data_down[:,:n_f]
    min_maxf=preprocessing.MinMaxScaler()
    data_down_n=min_maxf.fit_transform(data_down)
    df_down=np.c_[data_down_n,df_n[:,-1]]
    #降采样结束后，每个故障类型选取一定数量的数据集


    x_tr=np.empty(shape=(n_s,n_f))
    x_te=np.empty(shape=(n_s,n_f))
    y_tr=np.empty(shape=(n_s,))
    y_te=np.empty(shape=(n_s,))

    j=0
    for i in range(n_c):
        df_data=df_down[j:j+500,:-1]
        df_label=df_down[j:j+500,-1]
        x_train, x_test, y_train, y_test = train_test_split(df_data, df_label, test_size=0.5, random_state=0)
        x_tr=np.concatenate((x_tr,x_train[:n_s]))
        x_te=np.concatenate((x_te,x_test[:n_s]))
        y_tr=np.concatenate((y_tr,y_train[:n_s]))
        y_te=np.concatenate((y_te,y_test[:n_s]))
        j+=500
    x_train=np.array(x_tr[n_s:])
    x_test=np.array(x_te[n_s:])
    y_train=np.array(y_tr[n_s:])
    y_test=np.array(y_te[n_s:])

    return x_train,x_test,y_train,y_test

    
class AE(object):
    def __init__(self,x_train,y_train,x_test,y_test,
                     epochs_ae=100,epochs_c=100,batch_size_ae=50, activation_function='relu',
                     batch_size_c=50,lr_ae=0.6,decay=1e-6,lr_c=0.1,verbose=1,hidden_layer=[50,30,30,50],class_n=11):
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.epochs_ae=epochs_ae
        self.epochs_c=epochs_c
        self.batch_size_ae=batch_size_ae
        self.batch_size_c=batch_size_c
        self.lr_ae=lr_ae
        self.lr_c=lr_c
        self.decay=decay
        self.verbose=verbose
        self.activation_function = activation_function
        self.hidden_layer=hidden_layer
        self.class_n=class_n
        self.model = Sequential()
        self.index=int(len(self.hidden_layer)/2)

    def autoencoder(self):
        for i in range(0, len(self.hidden_layer)):
            if i == 0:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function,name='ae_'+str(i),
                                     input_dim=self.x_train.shape[1]))
            elif i >= 1:
                self.model.add(Dense(self.hidden_layer[i], activation=self.activation_function,name='ae_'+str(i)))
        self.model.add(Dense(self.x_train.shape[1], activation=self.activation_function))
        print(self.model.summary())
        adam_ae=optimizers.Adam(lr=self.lr_ae,decay=self.decay)
        self.model.compile(optimizer=adam_ae,loss='mse')
        self.model.fit(self.x_train,self.x_train,epochs=self.epochs_ae,batch_size=self.batch_size_ae,shuffle=True)
        print('autoencoder training finish.')
        self.model.save(r'E:\analogdata\ae.h5')
        self.model.save_weights(r'E:\analogdata\ae_weights.h5')
    def featureextraction(self):
        my_model=load_model(r'E:\analogdata\classify.h5')
        encoder=Model(inputs=my_model.input,outputs=my_model.get_layer(index=self.index).output)
        x_train_Dense=encoder.predict(self.x_train)
        x_test_Dense=encoder.predict(self.x_test)
        np.save(r'E:\analogdata\x_train_Dense.npy',x_train_Dense)
        np.save(r'E:\analogdata\x_test_Dense.npy',x_test_Dense)
        return x_train_Dense,x_test_Dense
    def classify(self):
        newmodel=Sequential()
        for i in range(0,self.index):
            if i==0:
                newmodel.add(Dense(self.hidden_layer[i],activation=self.activation_function,name='ae_'+str(i),
                                   input_dim=self.x_train.shape[1]))
            elif i>=1:
                newmodel.add(Dense(self.hidden_layer[i], activation=self.activation_function,name='ae_'+str(i)))
        newmodel.load_weights(r'E:\analogdata\ae_weights.h5',by_name=True)
        newmodel.add(Dense(self.class_n,activation='softmax',name='output'))
        print('分类器结构')
        print(newmodel.summary())
        adam_c=optimizers.Adam(lr=self.lr_c,decay=self.decay)
        newmodel.compile(optimizer=adam_c,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
        newmodel.fit(self.x_train,self.y_train,epochs=self.epochs_c,batch_size=self.batch_size_c,shuffle=True)
        score=newmodel.evaluate(self.x_test,self.y_test)
        newmodel.save(r'E:\analogdata\classify.h5')
        print(score)           
        return score
def showpic(x_test_Dense,y_test,n_w=2,n_c=11):
    r = random.randint(0, 1000)
    #TSNE降维
    tsne = TSNE(n_components=n_w).fit_transform(x_test_Dense)
    #保存降维后的数组
    x_test_f=np.c_[tsne,y_test]
    data1=pd.DataFrame(x_test_f)
    data_name=str(r)+'.csv'
    data1.to_csv(r'D:\Cadence\SPB_Data\analog circuit code\fault diagnosis\autoencoder'+data_name)
    col_list=['r','y','g','k','b','c','m','brown','orange','navy','chocolate','greenyellow','pink','aquamarine']
    shape_list=['3','v','s','p','*','h','+','x','D','o','<',',','.']
    img_name=str(r)+'.png'
    plt.figure(figsize=(5,5),dpi=200)
    if n_w==2:
       for label in range(n_c):
           index=np.argwhere(y_test==label)
           index_l=index.flatten()
           f_show=tsne[index_l]
           plt.scatter(f_show[:,0],f_show[:,1],c=col_list[label],marker=shape_list[label],s =10)

       plt.savefig(r'D:\Cadence\SPB_Data\analog circuit code\fault diagnosis\autoencoder\pic'+img_name)    
       plt.show()
    if n_w==3:
        plt.figure(figsize=(5,5),dpi=200)
        ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程
        for label in range(n_c):
            index=np.argwhere(y_test==label)
            index_l=index.flatten()
            f_show=tsne[index_l]
            ax.scatter(f_show[:,0],f_show[:,1],f_show[:,2],c=col_list[label],marker=shape_list[label],s = 5)
        plt.savefig(r'D:\Cadence\SPB_Data\analog circuit code\fault diagnosis\autoencoder\pic'+img_name)        
        plt.show()
def TBAE(x_train,x_test,y_train_sc,y_test_sc,p_ae=1,p_c=10,class_n=11,hidden_layer=[50,30],epoc=100,batch_s=50, 
         lr_a=0.6,decay_a=0.2,verbose=1):
    input_ae=Input(shape=(x_train.shape[1],))
#编码
    encoded=Dense(hidden_layer[0],activation='relu',name='encoded_hidden1')(input_ae)
    encoder_output=Dense(hidden_layer[1],activation='relu',name='encoded_hidden2')(encoded)
#解码
    decoded1=Dense(hidden_layer[0],activation='relu',name='decoded_hidden1')(encoder_output)
    #decoded1=Dense(hidden_layer[1],activation='relu',name='decoded_hidden1')(encoder_output)
    #decoded2=Dense(hidden_layer[0],activation='relu',name='decoded_hidden2')(decoded1)
    decoded_output=Dense(x_train.shape[1],activation='relu',name='decoded_output')(decoded1)
    ca=Dense(class_n,activation='softmax',name='class_out')(encoder_output)
    mymodel=Model(inputs=input_ae,outputs=[decoded_output,ca])
    print(mymodel.summary())   
    adam=optimizers.Adam(lr=lr_a,decay=decay_a)
    mymodel.compile(optimizer=adam,loss={'decoded_output':'mse','class_out':'categorical_crossentropy'},loss_weights={'decoded_output':p_ae,'class_out':p_c},metrics={'class_out':'categorical_accuracy'})
    history = LossHistory()
    history1=mymodel.fit(x_train,{'decoded_output':x_train,'class_out':y_train_sc},epochs=epoc,batch_size=batch_s,callbacks=[history])
    history_dict = history1.history
    print('history参数:',history_dict.keys())

    #绘制acc-loss曲线
    accy=history1.history['class_out_categorical_accuracy']
    lossy = history1.history['class_out_loss']
    np_accy = np.array(accy).reshape((1,len(accy))) #reshape是为了能够跟别的信息组成矩阵一起存储
    np_lossy = np.array(lossy).reshape((1,len(lossy)))
    acc_out = np.concatenate((np_accy,np_lossy))
    cs2 = pd.DataFrame(acc_out)
    cs2.to_csv(r'E:\analogdata\TBAE result\accloss.csv')
    print("保存文件成功")

    history.on_epoch_end(100)
    history.loss_plot('epoch')
    print('autoencoder training finish.')

    score=mymodel.evaluate(x_test,{'decoded_output':x_test,'class_out':y_test_sc})
    r=score[3]
    weight_name=str(r)+'.h5'
    #mymodel.save(r'E:\analogdata\TBAE result\TBAE_model.h5')
    mymodel.save_weights(r'E:\analogdata\TBAE result'+weight_name)
    print(score)
    return score
def TBAE_featureextraction(x_train,x_test,hidden_layer=[50,30],filename=r'E:\analogdata\TBAE_weights.h5'):
       input_fe=Input(shape=(x_train.shape[1],))
       if len(hidden_layer)==2:
          encoded_fe=Dense(hidden_layer[0],activation='relu',name='encoded_hidden1')(input_fe)
          encoder_output=Dense(hidden_layer[1],activation='relu',name='encoded_hidden2')(encoded_fe) 
       if len(hidden_layer)==1:
          encoder_output=Dense(hidden_layer[0],activation='relu',name='encoded_hidden1')(input_fe)           
       TBAE_fe=Model(inputs=input_fe,outputs=encoder_output)
       TBAE_fe.load_weights(filename,by_name=True)
       print('特征提取网络结构')
       print(TBAE_fe.summary())   
       x_train_Dense=TBAE_fe.predict(x_train)
       x_test_Dense=TBAE_fe.predict(x_test)
       x_train_D=pd.DataFrame(x_train_Dense)
       x_test_D=pd.DataFrame(x_test_Dense)
       x_train_D.to_csv(r'E:\论文写作\模拟电路论文构思\图片\最终版图片\x_train_TBAE_D.csv')
       x_test_D.to_csv(r'E:\论文写作\模拟电路论文构思\图片\最终版图片\x_test_TBAE_D.csv')
       return x_train_Dense,x_test_Dense
class RBM(object):
    def __init__(self, input_size, output_size,epochs=50,lr=0.2,batchsize=50):
        # Defining the hyperparameters
        self._input_size = input_size  # Size of input
        self._output_size = output_size  # Size of output
        self.epochs = epochs  # Amount of training iterations
        self.learning_rate = lr  # The step used in gradient descent
        self.batchsize = batchsize  # The size of how much data will be used for training per sub iteration

        # Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32)  # Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float32)  # Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float32)  # Creates and initializes the visible biases with 0

    # Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    # Training method for the model
    def train(self, X):
        # Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size],
                         np.float32)  # Creates and initializes the weights with 0
        prv_hb = np.zeros([self._output_size], np.float32)  # Creates and initializes the hidden biases with 0
        prv_vb = np.zeros([self._input_size], np.float32)  # Creates and initializes the visible biases with 0

        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])

        # Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        # Update learning rates for the layers
        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb + self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        # Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))

        # Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # For each epoch
            for epoch in range(self.epochs):
                # For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize), range(self.batchsize, len(X), self.batchsize)):
                    batch = X[start:end]
                    # Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    # Create expected output for our DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        w=tf.cast(_w,tf.float64)
        
        _hb = tf.constant(self.hb)
        b=tf.cast(_hb,tf.float64)
        out = tf.nn.sigmoid(tf.matmul(input_X,w) + b)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
class NN(object):

    def __init__(self, sizes, X, Y,x_test,y_test,lr=0.2,momentum=0.9,epoches=100,batchsize=50):
        # Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self._Xt=x_test
        self._Yt=y_test
        self.w_list = []
        self.b_list = []
        self._learning_rate = lr
        self._momentum = momentum
        self._epoches = epoches
        self._batchsize = batchsize
        input_size = X.shape[1]

        # initialization loop
        for size in self._sizes + [Y.shape[1]]:
            # Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))

            # Initialize weights through a random uniform distribution
            self.w_list.append(
                np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))

            # Initialize bias as zeroes
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    # load data from rbm
    def load_from_rbms(self, dbn_sizes, rbm_list):
        # Check if expected sizes are correct
        assert len(dbn_sizes) == len(self._sizes)

        for i in range(len(self._sizes)):
            # Check if for each RBN the expected sizes are correct
            assert dbn_sizes[i] == self._sizes[i]

        # If everything is correct, bring over the weights and biases
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    # Training method
    def train(self):
        # Create placeholders for input, weights, biases, output
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])

        # Define variables and activation functoin
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        # Define the cost function
        cost = tf.reduce_mean(tf.square(_a[-1] - y))

        # Define the training operation (Momentum Optimizer minimizing the Cost function)
        train_op = tf.train.MomentumOptimizer(
            self._learning_rate, self._momentum).minimize(cost)

        # Prediction operation
        predict_op = tf.argmax(_a[-1], 1)

        # Training Loop
        with tf.Session() as sess:
            # Initialize Variables
            sess.run(tf.global_variables_initializer())

            # For each epoch
            for i in range(self._epoches):

                # For each step
                for start, end in zip(
                        range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    # Run the training operation on the input data
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})

                for j in range(len(self._sizes) + 1):
                    # Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])

                    print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) == \
                                                                    sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))
            print("Test accuracy rating: " + str(np.mean(np.argmax(self._Yt, axis=1) == \
                                                                    sess.run(predict_op, feed_dict={_a[0]: self._Xt, y: self._Yt}))))
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))


    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        cs1 = pd.DataFrame(self.losses[loss_type])
        cs1.to_csv(r'E:\analogdata\TBAE result\losses.csv')

        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        
def TBDAE_train(x_train,x_test,y_train_sc,y_test_sc,class_n,n_drop=0.1,hidden_layer=[50,30],epoc=100,batch_s=50, 
         lr_a=0.6,decay_a=0.2,verbose=1):
    input_ae=Input(shape=(x_train.shape[1],))
    # Create the dropout layer
    dropout_layer = Dropout(n_drop)
    in_dropout = dropout_layer(input_ae)
#编码
    encoded=Dense(hidden_layer[0],activation='relu',name='encoded_hidden1')(in_dropout)
    encoder_output=Dense(hidden_layer[1],activation='relu',name='encoded_hidden2')(encoded)
#解码
    decoded1=Dense(hidden_layer[1],activation='relu',name='decoded_hidden1')(encoder_output)
    decoded2=Dense(hidden_layer[0],activation='relu',name='decoded_hidden2')(decoded1)
    decoded_output=Dense(x_train.shape[1],activation='relu',name='decoded_output')(decoded2)
    ca=Dense(class_n,activation='softmax',name='class_out')(encoder_output)
    mymodel=Model(inputs=input_ae,outputs=[decoded_output,ca])
    print(mymodel.summary())   
    adam=optimizers.Adam(lr=lr_a,decay=decay_a)
    mymodel.compile(optimizer=adam,loss={'decoded_output':'mse','class_out':'categorical_crossentropy'},loss_weights={'decoded_output':1.,'class_out':10},metrics={'class_out':'categorical_accuracy'})
    history = LossHistory()
    mymodel.fit(x_train,{'decoded_output':x_train,'class_out':y_train_sc},epochs=epoc,batch_size=batch_s,callbacks=[history])
    #绘制acc-loss曲线
    history.loss_plot('epoch')
    mymodel.save(r'E:\analogdata\TBDAE.h5')
    mymodel.save_weights(r'E:\analogdata\TBDAE_weights.h5')
    print('Dautoencoder training finish.')
    #score=mymodel.evaluate(x_test,{'decoded_output':x_test,'class_out':y_test_sc})
   # print(score)
   # return score
def TBDAE_test(x_train,x_test,y_test_sc,y_test,class_n,hidden_layer=[50,30]):
       input_fe=Input(shape=(x_train.shape[1],))
#编码
       encoded_fe=Dense(hidden_layer[0],activation='relu',name='encoded_hidden1')(input_fe)
       encoder_output=Dense(hidden_layer[1],activation='relu',name='encoded_hidden2')(encoded_fe) 
       ca=Dense(class_n,activation='softmax',name='class_out')(encoder_output)
       TBDAE_f=Model(inputs=input_fe,outputs=ca)
       TBDAE_f.load_weights(r'E:\analogdata\TBDAE_weights.h5',by_name=True)
       print('测试模型网络结构')
       print(TBDAE_f.summary())   
       y_test_p=TBDAE_f.predict(x_test)
       y_test_pl = [np.argmax(one_hot) for one_hot in y_test_p]

       print(accuracy_score(y_test,y_test_pl))
'''
单隐层自编码器
'''    
def TBAE_ol(x_train,x_test,y_train_sc,y_test_sc,class_n,hidden_layer=[50],epoc=100,batch_s=50, 
         lr_a=0.6,decay_a=0.2,verbose=1):
    input_ae=Input(shape=(x_train.shape[1],))
#编码
    encoder_output=Dense(hidden_layer[0],activation='relu',name='encoded_hidden1')(input_ae)
    
#解码
    decoded_output=Dense(x_train.shape[1],activation='relu',name='decoded_output')(encoder_output)
    ca=Dense(class_n,activation='softmax',name='class_out')(encoder_output)
    mymodel=Model(inputs=input_ae,outputs=[decoded_output,ca])
    print(mymodel.summary())   
    adam=optimizers.Adam(lr=lr_a,decay=decay_a)
    mymodel.compile(optimizer=adam,loss={'decoded_output':'mse','class_out':'categorical_crossentropy'},loss_weights={'decoded_output':3.,'class_out':6},metrics={'class_out':'categorical_accuracy'})
    history = LossHistory()
    history1=mymodel.fit(x_train,{'decoded_output':x_train,'class_out':y_train_sc},epochs=epoc,batch_size=batch_s,callbacks=[history])
    history_dict = history1.history
    print('history参数:',history_dict.keys())

    #绘制acc-loss曲线
    accy=history1.history['class_out_categorical_accuracy']
    lossy = history1.history['class_out_loss']
    np_accy = np.array(accy).reshape((1,len(accy))) #reshape是为了能够跟别的信息组成矩阵一起存储
    np_lossy = np.array(lossy).reshape((1,len(lossy)))
    acc_out = np.concatenate((np_accy,np_lossy))
    cs2 = pd.DataFrame(acc_out)
    cs2.to_csv(r'E:\analogdata\TBAE result\accloss.csv')
    print("保存文件成功")

    history.on_epoch_end(100)
    history.loss_plot('epoch')
    mymodel.save(r'E:\analogdata\TBAE.h5')
    mymodel.save_weights(r'E:\analogdata\TBAE_weights.h5')
    print('autoencoder training finish.')
    score=mymodel.evaluate(x_test,{'decoded_output':x_test,'class_out':y_test_sc})
    print(score)
    return score

