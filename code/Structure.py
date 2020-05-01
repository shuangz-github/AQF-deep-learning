# -*- coding: utf-8 -*-
#%%
# load packages
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#%%
# network parameters
N_assets = 50
n_asset_index = 2
n_macro_index = 8
# n_Moment_RNN = 4

#%%
# State RNN
inputs = keras.Input(shape=(n_macro_index))
n_macro_info = 4   # output dimension
n_samples = tf.shape(inputs)[0]
inputs0 = tf.reshape(inputs,[1,n_samples,n_macro_index]) # reshape it as a sequence

State_RNN1 = layers.LSTM(n_macro_info,return_sequences = True)
outputs0 = State_RNN1(inputs0)

outputs = tf.reshape(outputs0,[n_samples,n_macro_info])
State_RNN = keras.Model(inputs=inputs, outputs=outputs, name = 'State_RNN')
State_RNN.summary()
# inputs = keras.Input(shape=(n_macro_index,))
# outputs = State_RNN(inputs)
# test = keras.Model(inputs = inputs, outputs = outputs, name = 'test')
# test.summary()

#%%
# State FFN model
inputs = keras.Input(shape = (n_asset_index + n_macro_info))
State_FFN1 = layers.Dense(32, activation = 'relu')
State_FFN2 = layers.Dense(32, activation = 'relu')
State_FFN3 = layers.Dense(1, activation = 'linear')
outputs = State_FFN1(inputs)
outputs = State_FFN2(outputs)
outputs = State_FFN3(outputs)
State_FFN = keras.Model(inputs=inputs, outputs=outputs, name = 'State_FFN')
State_FFN.summary()

#%%
# Global State FFN model

inputs1 = keras.Input(shape = (N_assets, n_asset_index))
n_samples1 = tf.shape(inputs1)[0]
inputs10 = tf.reshape(inputs1,[n_samples1*N_assets,n_asset_index])

inputs2 = keras.Input(shape = (n_macro_info))
n_samples2 = tf.shape(inputs2)[0]
inputs20 = tf.tile(inputs2,[N_assets,1])

inputs = [inputs1, inputs2]

inputs0 = tf.concat([inputs10,inputs20],1)
outputs0 = State_FFN(inputs0)
outputs = tf.reshape(outputs0,[n_samples1,N_assets])

Global_State_FFN = keras.Model(inputs=inputs, outputs=outputs, name = 'Global_State_FFN')
Global_State_FFN.summary()

#%%
# State 
macro_index = keras.Input(shape=(n_macro_index))
asset_index = keras.Input(shape=(N_assets, n_asset_index))
R = keras.Input(shape=(N_assets))
inputs = [macro_index, asset_index, R]
macro_info = State_RNN(macro_index)
w = Global_State_FFN([asset_index, macro_info])
M = tf.multiply(R,w)
M = tf.reduce_sum(M, 1)
M = 1 - M
State = keras.Model(inputs=inputs, outputs=M, name = 'State')
State.summary()
# output has shape (time_steps, 1)

#%%
# Moment RNN
inputs = keras.Input(shape=(n_macro_index))
n_macro_info = 4   # Moment output dimension
n_samples = tf.shape(inputs)[0]
inputs0 = tf.reshape(inputs,[1,n_samples,n_macro_index]) # reshape it as a sequence
Moment_RNN1 = layers.LSTM(n_macro_info,return_sequences = True)
outputs0 = Moment_RNN1(inputs0)
outputs = tf.reshape(outputs0,[n_samples,n_macro_info])
Moment_RNN = keras.Model(inputs=inputs, outputs=outputs, name = 'Moment_RNN')
Moment_RNN.summary()

#%%
# Moment FFN model
inputs = keras.Input(shape = (n_asset_index + n_macro_info))
Moment_FFN1 = layers.Dense(32, activation = 'relu')
Moment_FFN2 = layers.Dense(32, activation = 'relu')
Moment_FFN3 = layers.Dense(1, activation = 'linear')
outputs = Moment_FFN1(inputs)
outputs = Moment_FFN2(outputs)
outputs = Moment_FFN3(outputs)
Moment_FFN = keras.Model(inputs=inputs, outputs=outputs, name = 'Moment_FFN')
State_FFN.summary()

#%%
# Global Moment FFN model

inputs1 = keras.Input(shape = (N_assets, n_asset_index))
n_samples1 = tf.shape(inputs1)[0]
inputs10 = tf.reshape(inputs1,[n_samples1*N_assets,n_asset_index])

inputs2 = keras.Input(shape = (n_macro_info))
n_samples2 = tf.shape(inputs2)[0]
inputs20 = tf.tile(inputs2,[N_assets,1])

inputs = [inputs1, inputs2]

inputs0 = tf.concat([inputs10,inputs20],1)
outputs0 = Moment_FFN(inputs0)
outputs = tf.reshape(outputs0,[n_samples1,N_assets])

Global_Moment_FFN = keras.Model(inputs=inputs, outputs=outputs, name = 'Global_Moment_FFN')
Global_Moment_FFN.summary()

#%%
# Moment
macro_index = keras.Input(shape=(n_macro_index))
asset_index = keras.Input(shape=(N_assets, n_asset_index))
inputs = [macro_index, asset_index]
macro_info = Moment_RNN(macro_index)
g = Global_Moment_FFN([asset_index, macro_info])
# output has shape (time_steps, N_assets)
Moment = keras.Model(inputs=inputs, outputs=g, name = 'Moment')

Moment.summary()

#%%
# Combined 
macro_index = keras.Input(shape=(n_macro_index))
asset_index = keras.Input(shape=(N_assets, n_asset_index))
R = keras.Input(shape=(N_assets))
print(macro_index.shape, asset_index.shape, R.shape)


inputs = [macro_index, asset_index, R]
M = State([macro_index, asset_index, R])
samples = tf.shape(M)[0]
M = tf.reshape(M,[samples,1])
M_tiled = tf.tile(M,[1,N_assets])

inputs0 = [macro_index, asset_index]
g = Moment(inputs0)

outputs = tf.multiply(g,R)
outputs = tf.multiply(outputs,M_tiled)

Combined = keras.Model(inputs=inputs, outputs=M_tiled, name = 'Combined')
Combined.summary()
# output has shape (time_steps, 1)
# keras.utils.plot_model(Combined, 'my_first_model.png')

#%%
# test
# macro_index = keras.Input(shape=(n_macro_index))
# asset_index = keras.Input(shape=(N_assets, n_asset_index))
R = keras.Input(shape=(N_assets))
# inputs = [macro_index, asset_index, R]
# outputs = [macro_index, asset_index, R]
outputs = keras.layers.Dense(N_assets)(R)
test = keras.Model(inputs=R, outputs=outputs, name = 'test')
# test.summary()

#%%
# Numerical Experiments
N_assets = 50
T_train = 25
T_valid = 10
T_test = 25
T = 60
n_asset_index = 2
n_macro_index = 8
np.random.seed(7)
beta = np.random.normal(size = (T, N_assets)) * np.random.normal(size = (T, N_assets))
F = 1 + np.random.normal(size = (T,1))
F = np.tile(F,(1, N_assets))
epsilon = np.random.normal(size = (N_assets,T))*0.5
R = beta*F #+ epsilon
macro_index = np.zeros(shape = (T, n_macro_index))
asset_index = np.zeros(shape = (T, N_assets, n_asset_index))
print(macro_index.shape, asset_index.shape, R.shape)
x_train = [macro_index, asset_index, R]
y_train = np.zeros(shape = (T, N_assets))

#%%
# 这里的compile随便写的compile过程，为了看看整个model对不对，Loss需要改成L2的加权Loss
# 另外训练也要用GAN的方法分别minmax这样训练，朋友，加油
def custom_loss(y_true,y_pred):
    z = y_pred - y_true
    z_abs = tf.abs(z)
    tf.reduce_sum(z_abs, [0, 1])
    return z
Combined.compile(optimizer='adam',loss=custom_loss)

#%%
Combined.fit(x=x_train,y=np.zeros(shape= (T,N_assets)), epochs=100,verbose=True)

#%% data generate / imput 



#%% model construction
# GAN (min -max)
# GAN (min-mean)
# FNN
# LS
# EN (LS with regularation)

#%% measures
# Sharpe
# explained varaince
# cross sectional R2

#%%
# training, validation, testing
# fixed window
# expanding window
# rolling window










