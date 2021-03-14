import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure

# this function load one .cvs (a sequence)
def load_data(dataset, csv_folder='./UAV-Attack-Dataset/'):
    print(os.getcwd())
    idx_split = []
    if dataset == 'PX4-VTOL-SITL':
        csv_folder = './UAV-Attack-Dataset/PX4-VTOL-SITL/GPS-Spoofing/'
        data_file = os.path.join(csv_folder, '2020-08-02-15-31-27-vehicle1-cut.csv')
        anomalies = ['2015-01-27 00:00:00']
        idx_split = [0, 300]
        t_unit = '1 sec'
    elif dataset == 'PX4-QUAD-HITL':
        csv_folder_normal = './UAV-Attack-Dataset/PX4-QUAD-HITL/Normal/'
        csv_folder_spoofed = './UAV-Attack-Dataset/PX4-QUAD-HITL/GPS-Spoofing/'
        data_file_normal = os.path.join(csv_folder_normal, '2020-08-02-16-23-17-vehicle1.csv')
        data_file_spoofed = os.path.join(csv_folder_spoofed, '2020-08-02-18-57-48-vehicle1.csv')
        features = ["roll", "pitch", "yawRate"]
        #features = ["roll", "pitch"]
        #features = ["roll"]
        anomalies = [1152]#['2020-08-02 19:16:58']
        cut_N = [0, 1250]
        cut_S = [180, 1500]
        #idx_split = [0, 900]
        t_unit = '1 sec'
    elif dataset == 'PX4-QUAD-SITL':
        csv_folder = './UAV-Attack-Dataset/PX4-QUAD-SITL/GPS-Spoofing/'
        data_file = os.path.join(csv_folder, '2020-08-02-10-13-05-vehicle1-cut.csv')
        anomalies = ['2015-01-27 00:00:00']
        idx_split = [0, 3000]
        t_unit = '1 sec'
    elif dataset == 'PX4-TAIL-SITL':
        csv_folder = './UAV-Attack-Dataset/PX4-TAIL-SITL/GPS-Spoofing/'
        data_file = os.path.join(csv_folder, '2020-08-02-14-58-35-vehicle1-cut.csv')
        anomalies = ['2015-01-27 00:00:00']
        idx_split = [0, 3000]
        t_unit = '1 sec'
    elif dataset == 'PX4-PLANE-SITL':
        csv_folder = './UAV-Attack-Dataset/PX4-PLANE-SITL/GPS-Spoofing/'
        data_file = os.path.join(csv_folder, '2020-08-02-19-56-41-vehicle1-cut.csv')
        anomalies = ['2015-01-27 00:00:00']
        idx_split = [0, 3000]
        t_unit = '1 sec'
    elif dataset == 'PX4-H480-SITL':
        csv_folder = './UAV-Attack-Dataset/PX4-H480-SITL/GPS-Spoofing/'
        data_file = os.path.join(csv_folder, '2020-08-02-14-20-26-vehicle1-cut.csv')
        anomalies = ['2015-01-27 00:00:00']
        idx_split= [0, 3000]
        t_unit = '1 sec'
    
    t = []
    #readings = []
    idx_anomaly = []
    i = 0
    with open(data_file_normal) as csvfileN:
        df = pd.read_csv(csvfileN)
        readings_N = df[features].to_numpy()
        readings_N = readings_N[cut_N[0]:cut_N[1],]
        #readCSVNormal = csv.reader(csvfileN, delimiter=',')
        #readings = np.append(readings, readCSVNormal)
    
    with open(data_file_spoofed) as csvfileS:
        df = pd.read_csv(csvfileS)
        readings_S = df[features].to_numpy()
        readings_S = readings_S[cut_S[0]:cut_S[1],]
        
    readings = np.append(readings_N, readings_S, axis=0)
    t = np.linspace(0, readings.shape[0] - 1, readings.shape[0])
    t = np.asarray(t)
    readings = np.asarray(readings)
    idx_split = [0, readings_N.shape[0]]
    idx_anomaly = [anomalies[0] - cut_S[0] + readings_N.shape[0]]
    print("\nOriginal csv file contains {} timestamps.".format(t.shape))
    print("Processed time series contain {} readings.".format(readings.shape))
    print("Anomaly indices are {}".format(idx_anomaly))
    
    return t, t_unit, readings, idx_anomaly, idx_split, csv_folder

# This function plots a dataset with the train/test split and known anomalies
# Relies on helper function load_data()

def process_and_save_specified_dataset(dataset, y_scale=5, save_file=False):
    t, t_unit, readings, idx_anomaly, idx_split, save_dir = load_data(dataset)
    
    # print("readings: ", readings.shape)
    # split into training and test sets
    training = readings[idx_split[0]:idx_split[1]]
    t_train = t[idx_split[0]:idx_split[1]]

    readings_normalised = np.zeros(readings.shape, dtype=float)
    # print("shape: !!! ", training.shape)

    # normalise by training mean and std 
    train_m = training.mean(axis=0) #np.mean(training[:channel])
    train_std = training.std(axis=0) #np.std(training[:channel])
    print("\nTraining set mean is {}".format(train_m))
    print("Training set std is {}".format(train_std))
    # print("readings_normalised: ", readings_normalised[:,1])
    channels_num = len(train_m)
    for channel in range(channels_num):
        readings_normalised[:,channel] = (readings[:,channel] - train_m[channel]) / train_std[channel]
    
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
        np.savez(save_dir+dataset+'/data.npz', t=t, t_unit=t_unit, readings=readings, idx_anomaly=idx_anomaly,
                    idx_split=idx_split, training=training, test=test, train_m=train_m, train_std=train_std,
                    t_train=t_train, t_test=t_test, idx_anomaly_test=idx_anomaly_test)
        print("\nProcessed time series are saved at {}".format(save_dir+dataset+'.npz'))
    else:
        print("\nProcessed time series are not saved.")
    
    # plot the whole normalised sequence
    fig, axs = plt.subplots(channels_num + 1, 1, figsize=(18, 4), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    # print("readings_normalised: ", readings_normalised[:,channel])
    # axs = axs.ravel()
    # for i in range(4):
    for channel in range(channels_num):
        axs[channel].plot(t, readings_normalised[:,channel])
        if idx_split[0] == 0:
            axs[channel].plot(idx_split[1]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
        else:
            for i in range(2):
                axs[channel].plot(idx_split[i]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'b--')
        for j in range(len(idx_anomaly)):
            axs[channel].plot(idx_anomaly[j]*np.ones(20), np.linspace(-y_scale,y_scale,20), 'r--')
        #     axs.plot(data[:,1])
        axs[channel].grid(True)
        axs[channel].set_xlim(0, len(t))
        axs[channel].set_ylim(-y_scale, y_scale)
        axs[channel].set_xlabel("timestamp (every {})".format(t_unit))
        axs[channel].set_ylabel("normalised readings")
        axs[channel].set_title("{} dataset\n(normalised by train mean {:.2f} and std {:.2f})".format(dataset, train_m[channel], train_std[channel]))
        axs[channel].legend(('data', 'train test set split', 'anomalies'))
        
    plt.show()
    
    return t, readings_normalised

dataset = 'PX4-QUAD-HITL'
#idx_split = [0,3300]

t, readings_normalised = process_and_save_specified_dataset(dataset, save_file=True)#, idx_split)