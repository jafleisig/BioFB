import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.fft import fft, fftfreq
from scipy import stats
from scipy.io import savemat
import numpy as np
from sklearn.model_selection import train_test_split


files = ["breathing_filtered.csv",
         "JacquieData1_20211126.csv",
         "JacquieData2_20211126.csv",
         "JacquieDec1_1.csv",
         "JacquieDec1_2.csv",
         "RayDec1_1.csv"]

sample_rate = 2000  # Sample rate in Hz
time_step = 1.0/sample_rate  # Time between samples in seconds
freq_bin = 1  # Bin size for frequency
freq_max = 45  # Maximum frequency
t_bin = 5  # Time bin in seconds

stomach_features = None
chest_features = None
targets = None

for f in files:
    # Read in the csv and get each important channel
    data = pd.read_csv(f)
    stomach_data = data["Stomach"].tolist()
    chest_data = data["Chest"].tolist()
    events = data["Event"].tolist()

    # Filter out high frequency noise above 20 Hz
    butt = butter(10, 20, 'low', fs=sample_rate, analog=False, output='sos')
    stomach_data = sosfilt(butt, stomach_data)
    chest_data = sosfilt(butt, chest_data)
    N = len(stomach_data)
    # Go through each time bin, compute fft, add to features
    for i in range(0, N - t_bin*sample_rate, t_bin*sample_rate):
        # Get the data for this window
        stomach_sample = stomach_data[i: i + t_bin*sample_rate]
        chest_sample = chest_data[i: i + t_bin*sample_rate]
        events_sample = np.array(events[i: i + t_bin*sample_rate])

        # Perform fft on stomach and chest channels
        stomach_fft = fft(stomach_sample)
        chest_fft = fft(chest_sample)
        x_freq = fftfreq(N, time_step)[:N//2]
        stomach_power = np.abs(stomach_fft) ** 2
        chest_power = np.abs(chest_fft) ** 2

        # Truncate data at max frequency
        max_index = np.max(np.where(np.round(x_freq) == freq_max))
        x_freq = x_freq[:max_index]
        stomach_power = stomach_power[:max_index]
        chest_power = chest_power[:max_index]
        # Divide frequencies into 45 equal bins and average their powers
        num_bins = freq_max
        bin_size = round(len(stomach_power)/num_bins)
        stomach_features_sample = [None for j in range(num_bins)]
        chest_features_sample = [None for j in range(num_bins)]
        for b in range(num_bins):
            stomach_features_sample[b] = np.mean(stomach_power[b*bin_size:(b+1)*bin_size])
            chest_features_sample[b] = np.mean(chest_power[b * bin_size:(b+1) * bin_size])

        if stomach_features is not None:
            stomach_features = np.vstack((stomach_features, np.array(stomach_features_sample)))
            chest_features = np.vstack((chest_features, np.array(chest_features_sample)))
        else:
            stomach_features = np.array(stomach_features_sample)
            chest_features = np.array(chest_features_sample)

        this_event = stats.mode(events_sample)[0][0]
        if this_event == 0:  # Resting, incorrect
            this_label = [0, 1]
        elif this_event == 1:  # Chest breathing, incorrect
            this_label = [0, 1]
        elif this_event == 2:  # Diaphragmatic breathing, correct
            this_label = [1, 0]
        elif this_event == 3:  # Fast breathing, incorrect
            this_label = [0, 1]
        else:  # Motion artifact, incorrect (usually done similar to resting)
            this_label = [0, 1]

        if targets is not None:
            targets = np.vstack((targets, np.array(this_label)))
        else:
            targets = np.array(this_label)


# We now have all the data, let's normalize it and partition it into a training and testing set
features = np.hstack((stomach_features, chest_features))
# features = features/np.max(np.max(features))
features_train, features_test, targets_train, targets_test = train_test_split(features, targets,
                                                                              test_size=0.3,
                                                                              random_state=1)

savemat('training_data_truncated.mat', {'features_train': features_train, 'features_test': features_test,
        'targets_train': targets_train, 'targets_test': targets_test})







