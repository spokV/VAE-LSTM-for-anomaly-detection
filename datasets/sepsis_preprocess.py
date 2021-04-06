import os
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure
from scipy.interpolate import interp1d

path = './datasets/sepsis'
path_a = './datasets/sepsis/training_setB'
path_b = './datasets/sepsis/training_setA'
dataset = 'sepsis'
save_dir = './datasets/sepsis/'
channels_num = 6
y_scale = 10
save_file = True

int_factor = 1
target = np.empty((0,), float)
training = np.empty((0,channels_num), float)
columns = []
anomalies = []
test = np.empty((0,channels_num), float)
t_unit = 1
idx_anomaly_accu = 0
k = 0
interpolate = True

max_sample_length = 1000
if interpolate == True:
    int_factor = 3
    max_sample_length = max_sample_length * int_factor

#def process_and_save_specified_dataset(dataset, y_scale=5, save_file=False):
#    t, t_unit, readings, idx_anomaly, idx_split, save_dir = load_data(dataset)
def process_and_save_specified_dataset(readings, idx_anomaly, y_scale=5, save_file=False, plot=False):
    # print("readings: ", readings.shape)
    # split into training and test sets
    training = readings.values#readings[idx_split[0]:idx_split[1]]
    #t_train = t[idx_split[0]:idx_split[1]]
    t = np.linspace(0, training.shape[0] - 1, training.shape[0])
    t_int = np.linspace(0, training.shape[0] - 1, training.shape[0] * int_factor)
    t_unit = 1

    readings_normalised = np.zeros(training.shape, dtype=float)
    readings_normalised_int = np.zeros((training.shape[0] * int_factor, training.shape[1]), dtype=float)
    # print("shape: !!! ", training.shape)

    # normalise by training mean and std 
    train_m = training.mean(axis=0) #np.mean(training[:channel])
    train_std = training.std(axis=0) #np.std(training[:channel])
    #print("\nTraining set mean is {}".format(train_m))
    #print("Training set std is {}".format(train_std))
    # print("readings_normalised: ", readings_normalised[:,1])
    channels_num = len(train_m)
    for channel in range(channels_num):
        readings_normalised[:,channel] = (training[:,channel] - train_m[channel]) / train_std[channel]
        if interpolate == True:
            f = interp1d(t, readings_normalised[:,channel])
            readings_normalised_int[:,channel] = f(t_int)
    
    if interpolate == True:
        readings_normalised = readings_normalised_int
        t = t_int
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
            #for j in range(len(idx_anomaly)):
            axs[channel].plot(idx_anomaly*np.ones(20), np.linspace(-y_scale,y_scale,20), 'r--')
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

def plot_training_data(data, t, channels_num, is_train=True):
    fig, axs = plt.subplots(channels_num, 1, figsize=(18, 17), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    for channel in range(channels_num):
        axs[channel].plot(t, data[:,channel])
        """
        if idx_split[0] == 0:
            axs[channel].plot(idx_split[1]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
        else:
            for i in range(2):
                axs[channel].plot(idx_split[i]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
        """
        if is_train is False:
            for j in range(len(anomalies)):
                axs[channel].plot(anomalies[j]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'r--')
            #     axs.plot(data[:,1])
        
        axs[channel].grid(True)
        axs[channel].set_xlim(0, len(t))
        axs[channel].set_ylim(-y_scale, y_scale)
        axs[channel].set_xlabel("timestamp (every {})".format(t_unit))
        axs[channel].set_ylabel(columns[channel])
        #axs[channel].set_title("{} dataset (normalised by train mean {:.2f} and std {:.2f})".format(dataset, train_m[channel], train_std[channel]))
        #axs[channel].legend(('data', 'train test set split', 'anomalies'))

    plt.show()


for i in os.listdir(path_a):
    data = pd.read_csv(path_a+'/'+i,sep = '|')
    data.drop(['EtCO2','Fibrinogen', 'Unit1', 'Unit2', 'BaseExcess', 'DBP', 'Hct', 'Hgb', 'PTT', 
    'WBC', 'pH','HCO3','FiO2', 'PaCO2', 'Platelets', 'Magnesium',  'Phosphate',  'Potassium', 
    'Bilirubin_total',  'TroponinI','SaO2', 'AST','BUN', 'Alkalinephos', 'Bilirubin_direct','Glucose',
    'Gender', 'Age', 'ICULOS', 'HospAdmTime',
    'Lactate', 'Calcium',  'Chloride', 'Creatinine' ],axis = 1,inplace = True)

    #data.dropna(thresh=data.shape[1]*0.40,how='all',inplace = True)
    data.interpolate(axis=0, inplace=True)
    is_septic = True
    if data['SepsisLabel'].sum() == 0:
        is_septic = False
    
    data.dropna(inplace=True)
    idx_anomaly_local = data[data["SepsisLabel"]==1].index.values
    plot = False
    if idx_anomaly_local.size == 0:
        idx_anomaly_local = np.append(idx_anomaly_local, [0])
    else:
        idx_anomaly_local = idx_anomaly_local * int_factor
        plot = False
    columns = data.columns
    
    if len(data) > 100:
        #target = np.append(target, data["SepsisLabel"].values, axis=0)
        data.drop(['SepsisLabel'],axis = 1,inplace = True)
    
        t, readings_normalised = process_and_save_specified_dataset(data, idx_anomaly_local[0], y_scale=10, plot=plot)
        if is_septic is True:
            plot = False
            #test.append(readings_normalised)
            if test.shape[0] < max_sample_length:
                if k%3 == 0:
                    test = np.append(test, readings_normalised, axis=0)
                    idx_anomaly_accu += idx_anomaly_local[0]
                    anomalies.append(idx_anomaly_accu)
                    k += 1
        else:
            if k%3 != 0:
                test = np.append(test, readings_normalised, axis=0)
                idx_anomaly_accu += readings_normalised.shape[0]
                k+=1
            elif training.shape[0] < max_sample_length:
                training = np.append(training, readings_normalised, axis=0)
            else:
                break
            
            
def compose_final_data(training, test):
    pass

train_m = training.mean(axis=0) #np.mean(training[:channel])
train_std = training.std(axis=0)
print(training.shape)
print(test.shape)
t_train = np.linspace(0, training.shape[0] - 1, training.shape[0])
t_test = np.linspace(0, test.shape[0] - 1, test.shape[0])
channels_num = training.shape[1]
#target = np.atleast_2d(target).T
#plot_training_data(np.append(training, target, axis=1), channels_num + 1)
plot_training_data(training, t_train, channels_num, is_train=True)
plot_training_data(test, t_test, channels_num, is_train=False)

if save_file:
    #save_dir = './datasets/NAB-known-anomaly/'
    np.savez(save_dir+dataset+'.npz', t_unit=t_unit,
                training=training, test=test,
                t_train=t_train, t_test=t_test, idx_anomaly_test=anomalies)
    print("\nProcessed time series are saved at {}".format(save_dir+dataset+'.npz'))
else:
    print("\nProcessed time series are not saved.")