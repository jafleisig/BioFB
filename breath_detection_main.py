import pandas as pd
import visualization
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, butter, sosfilt
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import load
import serial
import time

ser = serial.Serial('COM4')

# Read the CSV
# breathing_filtered = pd.read_csv("JacquieData1_20211126.csv")
breathing_filtered = pd.read_csv("breathing_filtered.csv")
# breathing_filtered = pd.read_csv("RayDec1_2.csv")
# The original data is sampled at 2 kHz, this is way more than we need, so take only every 200 points (10 Hz)
breathing_filtered_sparse = breathing_filtered[0::200]

# Overall breathing data plot
stomach_data = breathing_filtered_sparse["Stomach"]  # Get stomach data using "Stomach" key
chest_data = breathing_filtered_sparse["Chest"]  # Get chest data using "Chest" key

# Filter out high frequency noise above 20 Hz
butt = butter(10, 20, 'low', fs=200, analog=False, output='sos')
stomach_data = sosfilt(butt, stomach_data)
chest_data = sosfilt(butt, chest_data)

sample_rate = 10  # Sample rate in Hz (which is now 10 since we've taken every 200th point in a 2 kHz sample
# Generate time point array since "Elapsed Time" in csv is strings, not floats
time_points = [float(i/sample_rate) for i in range(len(breathing_filtered_sparse))]  # Generate time points
time_step = 1.0/100  # Time between each data point
max_time = len(time_points)/time_step  # Max time of the entire recording
freq_max = 45

# Labels: 0 (regular), 1 (chest), 2 (diaphragm), 3 (fast breathing), 4 (low freq motion)
events = breathing_filtered_sparse["Event"]
event_times = visualization.get_event_times(events, time_points)
# Plot the data in the time domain for both channels
visualization.plot_csv(stomach_data, chest_data, time_points, event_times)

# Overall fft plot
stomach_fft = fft(stomach_data.tolist())  # fft from stomach channel
chest_fft = fft(chest_data.tolist())  # fft from chest channel
N = len(stomach_data)  # Number of data points
# Plot the fft for the entire time
# visualization.plot_fft(N, time_step, stomach_fft, chest_fft)

# Moving recordings
fft_fig, fft_ax = visualization.initialize_fft_frame()  # Initialize the figure
window_size = 5  # Width of the recording window in seconds
window_step = 1  # Step size in seconds (how far window moves in each iteration)

# Load in the classifier
clf = load('sklearn_nn.joblib')

# For i such that we start at 0 and go until the last window reaches the end of the data (about 120 iterations)
while True:
    motors = [None]  # List of indices that the motor starts vibrating
    nn_output = []
    for i in range(int(round(time_points[-1] - window_size)/window_step)):

        stomach_sample = stomach_data[i*window_step*sample_rate:(i*window_step+window_size)*sample_rate]  # Get stomach data within window
        chest_sample = chest_data[i*window_step*sample_rate:(i*window_step+window_size)*sample_rate]  # Get chest data within window
        stomach_fft = fft(stomach_sample.tolist())  # Do FFT on stomach sample
        chest_fft = fft(chest_sample.tolist())  # Do FFT on chest sample
        N = len(stomach_sample)  # Number of points within the sample

        # Peak analysis in frequency domain, if breathing rate is more than max breaths/minute, triple buzz
        max_bpm = 5
        max_amp_fft = np.argmax(np.abs(stomach_fft))
        stomach_power = np.abs(stomach_fft)**2
        x_freq = fftfreq(N, time_step)[:N // 2]  # x frequencies from online example
        max_freq = x_freq[max_amp_fft]
        peak_freq = x_freq[stomach_power.argmax()]*2

        # Get the fft power into frequency bins for input into neural network
        chest_power = np.abs(chest_fft)**2
        # Divide frequencies into 45 equal bins and average their powers
        num_bins = freq_max
        bin_size = round(len(stomach_power) / num_bins)
        stomach_features_sample = [None for j in range(num_bins)]
        chest_features_sample = [None for j in range(num_bins)]
        for b in range(num_bins):
            stomach_features_sample[b] = np.mean(stomach_power[b * bin_size:(b + 1) * bin_size])
            chest_features_sample[b] = np.mean(chest_power[b * bin_size:(b + 1) * bin_size])
        features_sample = np.hstack((np.array(stomach_features_sample), np.array(chest_features_sample)))

        # Predict the output using the classifier
        features_sample = features_sample.reshape(1, -1)
        nn_output = nn_output + clf.predict(features_sample).tolist()

        if nn_output[-1][0] == 0:  # NN says bad breathing
            # Peak analysis in time domain, if breathing rate is more than max breaths/minute, triple buzz
            # peaks = find_peaks(stomach_sample, distance=None)
            # rate = peaks[0].size/(len(stomach_sample)/200)
            if (peak_freq > max_bpm) and ((len(motors) == 1) or (abs(motors[-1]) < i-10)):
                motors = motors + [-i]
                ser.write('2'.encode('utf-8'))

            # If max power of chest is greater than 10 and there hasn't been a stimulus in the last 10 iterations
            elif (max(2.0 / N * np.abs(chest_fft[0:N // 2])) > 10) and ((len(motors) == 1) or (abs(motors[-1]) < i-10)):
                motors = motors + [i]  # Record this as a motor start time
                ser.write('1'.encode('utf-8'))

        # Plot the FFTs, time domain data, and motor pulses in the same plot
        visualization.plot_fft_frame(N, stomach_fft, chest_fft, x_freq, fft_ax, stomach_data, chest_data,
                                     time_points, event_times, window_step, window_size, i, motors, nn_output)



