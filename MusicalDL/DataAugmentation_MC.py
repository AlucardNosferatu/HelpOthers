# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:54:40 2019

@author: 16413
"""
import librosa as lr
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import fftpack,signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

def lowpass(input_array,cutoff=0.0055):
    b,a=signal.butter(8,cutoff,'lowpass')
    output_array=signal.filtfilt(b,a,np.ravel(input_array))
    output_array=np.expand_dims(output_array, axis=0)
    return output_array

def highpass(input_array,cutoff=0.0054):
    b,a=signal.butter(8,cutoff,'highpass')
    output_array=signal.filtfilt(b,a,np.ravel(input_array))
    output_array=np.expand_dims(output_array, axis=0)
    return output_array

def bandpass(input_array,upper=0.65,lower=0.64):
    b,a=signal.butter(8,[lower,upper],'bandpass')
    output_array=signal.filtfilt(b,a,np.ravel(input_array))
    output_array=np.expand_dims(output_array, axis=0)
    return output_array

def loadwav(index=0,tag='blues'):
    if(index<10):
        index='0'+str(index)
    else:
        index=str(index)
    filepath='Data_MusicClassification\\'+tag+'\\'+tag+'.000'+index+'.au'
    x,sr=lr.load(filepath)
    return x,sr

def processwav(x):
    wn = np.random.randn(len(x))
    x=np.where(x != 0.0, x + 0.02 * wn, 0.0)
    return x

def freq_ica(x,upper=0.3,lower=0.2):
    x_hp=highpass(x,cutoff=upper)
    x_bp=bandpass(x,upper=upper,lower=lower)
    x_lp=lowpass(x,cutoff=lower)
    x_fs=np.concatenate((x_hp, x_bp, x_lp), axis=0)
    x_fs=x_fs.T
    ica = FastICA(n_components=3)
    u = ica.fit_transform(x_fs)
    u = u.T
    return u,x_fs

for index in range(0,1):
    tag='blues'
    x,sr=loadwav(index=index,tag=tag)
    u,x_fs=freq_ica(x)
    u_df=pd.DataFrame(u)
    fs_df=pd.DataFrame(x_fs.T)
    
    u_df.loc[0,:].plot(figsize=(20,10))
    plt.show()
    u_df.loc[1,:].plot(figsize=(20,10))
    plt.show()
    u_df.loc[2,:].plot(figsize=(20,10))
    plt.show()
    wavfile.write("mp3\\test1.mp3",sr,u_df.loc[0,:].values)
    wavfile.write("mp3\\test2.mp3",sr,u_df.loc[1,:].values)
    wavfile.write("mp3\\test3.mp3",sr,u_df.loc[2,:].values)

#    x_p=processwav(x)
#    
#    if(index<10):
#        index='0'+str(index)
#    else:
#        index=str(index)
