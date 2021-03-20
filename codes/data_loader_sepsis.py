from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from scipy.interpolate import interp1d


class DataGenerator(BaseDataGenerator):
  def __init__(self, config):
    super(DataGenerator, self).__init__(config)
    # load data here: generate 3 state variables: train_set, val_set and test_set
    self.load_dataset(self.config['dataset'], self.config['y_scale'])

  def load_dataset(self, dataset, y_scale=6):
    data_dir = '../datasets/' + self.config['data_dir'] #NAB-known-anomaly/'
    #data = np.load(data_dir + dataset + '.npz')
    data = np.load(data_dir + self.config['filename'] + '.npz')

    training = data['training']
    t_train_new = np.linspace(0, data['training'].shape[0] - 1, data['training'].shape[0]*self.config['upsampling_factor'])
    training = np.pad(training, ((0,data['training'].shape[0]* (self.config['upsampling_factor'] - 1)),(0,0)), 'constant')
    for i in range(data['training'].shape[1]):
    #tr = data['training'][:,1]
        #print(training.shape)
        #print(data['t_train'].shape)
        inter_func = interp1d(data['t_train'], data['training'][:,i], kind='cubic')
        training[:,i] = inter_func(t_train_new)
        #print(tr2.shape)
    
    # slice training set into rolling windows
    n_train_sample = len(training)
    print("training: ", training)
    n_train_vae = n_train_sample - self.config['l_win'] + 1
    if self.config['n_channel'] == 1:
        rolling_windows = np.zeros((n_train_vae, self.config['l_win']))
    else:
        rolling_windows = np.zeros((n_train_vae, self.config['l_win'], training.shape[1]))
    print("training: ", rolling_windows.shape)
    for i in range(n_train_sample - self.config['l_win'] + 1):
      rolling_windows[i] = training[i:i + self.config['l_win']]

    # create VAE training and validation set
    idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
    if self.config['n_channel'] == 1:
        self.train_set_vae = dict(data=np.expand_dims(rolling_windows[idx_train], -1))
        self.val_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val], -1))
        self.test_set_vae = dict(data=np.expand_dims(rolling_windows[idx_val[:self.config['batch_size']]], -1))
    else:
        self.train_set_vae = dict(data=rolling_windows[idx_train])
        self.val_set_vae = dict(data=rolling_windows[idx_val])
        self.test_set_vae = dict(data=rolling_windows[idx_val[:self.config['batch_size']]])
    
    # create LSTM training and validation set
    for k in range(self.config['l_win']):
      n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
      n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
      if self.config['n_channel'] == 1:
          cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win']))
      else:
          cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win'], training.shape[1]))
      for i in range(n_train_lstm):
        if self.config['n_channel'] == 1:
            cur_seq = np.zeros((self.config['l_seq'], self.config['l_win']))
        else:
            cur_seq = np.zeros((self.config['l_seq'], self.config['l_win'], training.shape[1]))
        for j in range(self.config['l_seq']):
          # print(k,i,j)
          cur_seq[j] = training[k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]
        cur_lstm_seq[i] = cur_seq
      if k == 0:
        lstm_seq = cur_lstm_seq
      else:
        lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)

    n_train_lstm = lstm_seq.shape[0]
    idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
    if self.config['n_channel'] == 1:
        self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
        self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
    else:
        self.train_set_lstm = dict(data=lstm_seq[idx_train])
        self.val_set_lstm = dict(data=lstm_seq[idx_val])

  def plot_time_series(self, data, time, data_list):
    fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
    fig.subplots_adjust(hspace=.8, wspace=.4)
    axs = axs.ravel()
    for i in range(4):
      axs[i].plot(time / 60., data[:, i])
      axs[i].set_title(data_list[i])
      axs[i].set_xlabel('time (h)')
      axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
    savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')
  
  def upsample_test_samples(self, data, t):
    test = data
    t_test_new = np.linspace(0, data.shape[0] - 1, data.shape[0]*self.config['upsampling_factor'])
    test = np.pad(test, ((0,data.shape[0]* (self.config['upsampling_factor'] - 1)),(0,0)), 'constant')
    for i in range(data.shape[1]):
    #tr = data['training'][:,1]
        #print(training.shape)
        #print(data['t_train'].shape)
        inter_func = interp1d(t, data[:,i], kind='cubic')
        test[:,i] = inter_func(t_test_new)
    return test

