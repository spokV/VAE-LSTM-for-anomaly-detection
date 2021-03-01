import os
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure

path = './datasets/sepsis'
path_a = './datasets/sepsis/training_setB'
path_b = './datasets/sepsis/training_setA'
dataset = 'sepsis'
channels_num = 6
y_scale = 10
save_file = False

target = np.empty((0,), float)
training = np.empty((0,channels_num), float)
columns = []
t_unit = 1

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
    #print("\nTraining set mean is {}".format(train_m))
    #print("Training set std is {}".format(train_std))
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

def plot_training_data(training, channels_num):
    fig, axs = plt.subplots(channels_num, 1, figsize=(18, 17), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    for channel in range(channels_num):
        axs[channel].plot(t, training[:,channel])
        """
        if idx_split[0] == 0:
            axs[channel].plot(idx_split[1]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
        else:
            for i in range(2):
                axs[channel].plot(idx_split[i]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
        
        for j in range(len(idx_anomaly)):
            axs[channel].plot(idx_anomaly[j]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'r--')
        #     axs.plot(data[:,1])
        """
        axs[channel].grid(True)
        axs[channel].set_xlim(0, len(t))
        axs[channel].set_ylim(-y_scale, y_scale)
        axs[channel].set_xlabel("timestamp (every {})".format(t_unit))
        axs[channel].set_ylabel(columns[channel])
        #axs[channel].set_title("{} dataset (normalised by train mean {:.2f} and std {:.2f})".format(dataset, train_m[channel], train_std[channel]))
        #axs[channel].legend(('data', 'train test set split', 'anomalies'))

    plt.show()


for i in os.listdir(path_a):
    anomalies = []
    data = pd.read_csv(path_a+'/'+i,sep = '|')
    data.drop(['EtCO2','Fibrinogen', 'Unit1', 'Unit2', 'BaseExcess', 'DBP', 'Hct', 'Hgb', 'PTT', 
    'WBC', 'pH','HCO3','FiO2', 'PaCO2', 'Platelets', 'Magnesium',  'Phosphate',  'Potassium', 
    'Bilirubin_total',  'TroponinI','SaO2', 'AST','BUN', 'Alkalinephos', 'Bilirubin_direct','Glucose',
    'Gender', 'Age', 'ICULOS', 'HospAdmTime',
    'Lactate', 'Calcium',  'Chloride', 'Creatinine' ],axis = 1,inplace = True)

    #data.dropna(thresh=data.shape[1]*0.40,how='all',inplace = True)
    data.interpolate(axis=0, inplace=True)
    La_1 = data['SepsisLabel'].sum()
    """
    if La_1:
        y_train.append(1)
    else:
        y_train.append(0)
    """
    #filter = data["SepsisLabel"]==1
    #data = data.apply(lambda x: x.fillna(x.median()),axis=0)
    #data = data.fillna(0)
    data.dropna(inplace=True)
    idx_anomaly = data[data["SepsisLabel"]==1].index.values
    columns = data.columns
    
    if len(data) > 100:
        target = np.append(target, data["SepsisLabel"].values, axis=0)
        data.drop(['SepsisLabel'],axis = 1,inplace = True)
        
        plot = False
        if La_1 != 0:
            plot = False
            anomalies.append(idx_anomaly[0])
        t, readings_normalised = process_and_save_specified_dataset(data, anomalies, y_scale=10, plot=plot)
        if La_1 >= 0:
            training = np.append(training, readings_normalised, axis=0)
            if training.shape[0] > 1000:
                break

train_m = training.mean(axis=0) #np.mean(training[:channel])
train_std = training.std(axis=0)
print(training.shape)
t = np.linspace(0, training.shape[0] - 1, training.shape[0])
channels_num = training.shape[1]
target = np.atleast_2d(target).T
plot_training_data(np.append(training, target, axis=1), channels_num + 1)

if save_file:
    #save_dir = './datasets/NAB-known-anomaly/'
    np.savez(save_dir+dataset+'.npz', t=t, t_unit=t_unit, readings=readings, idx_anomaly=idx_anomaly,
                idx_split=idx_split, training=training, test=test, train_m=train_m, train_std=train_std,
                t_train=t_train, t_test=t_test, idx_anomaly_test=idx_anomaly_test)
    print("\nProcessed time series are saved at {}".format(save_dir+dataset+'.npz'))
else:
    print("\nProcessed time series are not saved.")