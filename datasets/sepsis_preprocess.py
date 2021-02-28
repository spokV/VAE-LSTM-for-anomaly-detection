import os
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM

path = './datasets/sepsis'
path_a = './datasets/sepsis/training_setB'
path_b = './datasets/sepsis/training_setA'
dataset = 'sepsis'

y_train = []
X_train = []

#def process_and_save_specified_dataset(dataset, y_scale=5, save_file=False):
#    t, t_unit, readings, idx_anomaly, idx_split, save_dir = load_data(dataset)
def process_and_save_specified_dataset(readings, idx_anomaly, y_scale=5, save_file=False, plot=False):
    # print("readings: ", readings.shape)
    # split into training and test sets
    training = readings.values#readings[idx_split[0]:idx_split[1]]
    #t_train = t[idx_split[0]:idx_split[1]]
    t = np.linspace(0, training.shape[0] - 1, training.shape[0])
    t_unit = 1

    readings_normalised = np.zeros(training.shape, dtype=float)
    # print("shape: !!! ", training.shape)

    # normalise by training mean and std 
    train_m = training.mean(axis=0) #np.mean(training[:channel])
    train_std = training.std(axis=0) #np.std(training[:channel])
    print("\nTraining set mean is {}".format(train_m))
    print("Training set std is {}".format(train_std))
    # print("readings_normalised: ", readings_normalised[:,1])
    channels_num = len(train_m)
    for channel in range(channels_num):
        readings_normalised[:,channel] = (training[:,channel] - train_m[channel]) / train_std[channel]
    
    """
    training = readings_normalised[idx_split[0]:idx_split[1]]
    if idx_split[0] == 0:
        test = readings_normalised[idx_split[1]:]
        t_test = t[idx_split[1]:] - idx_split[1]
        idx_anomaly_test = np.asarray(idx_anomaly) - idx_split[1]
    else:
        test = [readings_normalised[:idx_split[0]], readings_normalised[idx_split[1]:]]
        t_test = [t[:idx_split[0]], t[idx_split[1]:] - idx_split[1]]
        idx_anomaly_split = np.squeeze(np.argwhere(np.asarray(idx_anomaly)>idx_split[0]))
        idx_anomaly_test = [np.asarray(idx_anomaly[:idx_anomaly_split[0]]), 
                            np.asarray(idx_anomaly[idx_anomaly_split[0]:]) - idx_split[1]]
    print("Anomaly indices in the test set are {}".format(idx_anomaly_test))
    
    if save_file:
        #save_dir = './datasets/NAB-known-anomaly/'
        np.savez(save_dir+dataset+'.npz', t=t, t_unit=t_unit, readings=readings, idx_anomaly=idx_anomaly,
                    idx_split=idx_split, training=training, test=test, train_m=train_m, train_std=train_std,
                    t_train=t_train, t_test=t_test, idx_anomaly_test=idx_anomaly_test)
        print("\nProcessed time series are saved at {}".format(save_dir+dataset+'.npz'))
    else:
        print("\nProcessed time series are not saved.")
    """
    # plot the whole normalised sequence
    if plot==True:
        fig, axs = plt.subplots(channels_num, 1, figsize=(18, 17), edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.4)
        # print("readings_normalised: ", readings_normalised[:,channel])
        # axs = axs.ravel()
        # for i in range(4):
        for channel in range(channels_num):
            axs[channel].plot(t, readings_normalised[:,channel])
            """
            if idx_split[0] == 0:
                axs[channel].plot(idx_split[1]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
            else:
                for i in range(2):
                    axs[channel].plot(idx_split[i]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
            """
            for j in range(len(idx_anomaly)):
                axs[channel].plot(idx_anomaly[j]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'r--')
            #     axs.plot(data[:,1])
            
            axs[channel].grid(True)
            axs[channel].set_xlim(0, len(t))
            axs[channel].set_ylim(-y_scale, y_scale)
            axs[channel].set_xlabel("timestamp (every {})".format(t_unit))
            axs[channel].set_ylabel(readings.columns[channel])
            axs[channel].set_title("{} dataset (normalised by train mean {:.2f} and std {:.2f})".format(dataset, train_m[channel], train_std[channel]))
            #axs[channel].legend(('data', 'train test set split', 'anomalies'))
        
        plt.show()
    
    return t, readings_normalised

for i in os.listdir(path_a):
    anomalies = []
    data = pd.read_csv(path_a+'/'+i,sep = '|')
    data.drop(['EtCO2','Fibrinogen', 'Unit1', 'Unit2', 'BaseExcess', 'DBP', 'Hct', 'Hgb', 'PTT', 
    'WBC', 'pH','HCO3','FiO2', 'PaCO2', 'Platelets', 'Magnesium',  'Phosphate',  'Potassium', 
    'Bilirubin_total',  'TroponinI','SaO2', 'AST','BUN', 'Alkalinephos', 'Bilirubin_direct','Glucose',
    'Gender', 'Age', 'ICULOS', 'HospAdmTime',
    'Lactate', 'Calcium',  'Chloride', 'Creatinine' ],axis = 1,inplace = True)

    #data.dropna(thresh=data.shape[1]*0.40,how='all',inplace = True)
    La_1 = data['SepsisLabel'].sum()
    if La_1:
        y_train.append(1)
    else:
        y_train.append(0)
    
    #filter = data["SepsisLabel"]==1
    #data.drop(['SepsisLabel'],axis = 1,inplace = True)
    data = data.apply(lambda x: x.fillna(x.median()),axis=0)
    data = data.fillna(0)
    idx_anomaly = data[data["SepsisLabel"]==1].index.values
    
    if len(data) > 100:
        
    #if len(data) < 40:
    #    Pad = pd.DataFrame({'HR':0.0 ,'O2Sat':0.0, 'Temp':0.0 , 'SBP':0.0, 'MAP':0.0, 'Resp':0.0, 'Age':0.0, 'Gender': 0 ,'HospAdmTime':0.0, 'ICULOS':0}, index =[item for item in range(0,40-len(data))])
    #    data = pd.concat([Pad, data]).reset_index(drop = True)
    #elif len(data) >40:
    #    data = data[len(data)-40::1]
    #data = data.values
        
        plot = False
        if La_1 != 0:
            plot = True
            anomalies.append(idx_anomaly[0])
        t, readings_normalised = process_and_save_specified_dataset(data, anomalies, y_scale=10, plot=plot)
        X_train.append(readings_normalised)

print(X_train.shape)