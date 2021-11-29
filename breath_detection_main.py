import pandas as pd
import visualization
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import numpy as np

# Read the CSV
breathing_filtered = pd.read_csv("breathing_filtered.csv")
# The original data is sampled at 2 kHz, this is way more than we need, so take only every 200 points (10 Hz)
breathing_filtered_sparse = breathing_filtered[0::200]

# Overall breathing data plot
stomach_data = breathing_filtered_sparse["Stomach"]  # Get stomach data using "Stomach" key
chest_data = breathing_filtered_sparse["Chest"]  # Get chest data using "Chest" key
sample_rate = 10  # Sample rate in Hz (which is now 10 since we've taken every 200th point in a 2 kHz sample
# Generate time point array since "Elapsed Time" in csv is strings, not floats
time_points = [float(i/sample_rate) for i in range(len(breathing_filtered_sparse))]  # Generate time points
time_step = 1.0/100  # Time between each data point
max_time = len(time_points)/time_step  # Max time of the entire recording
# Plot the data in the time domain for both channels
# visualization.plot_csv(stomach_data, chest_data, time_points)

# Overall fft plot
stomach_fft = fft(stomach_data.tolist())  # fft from stomach channel
chest_fft = fft(chest_data.tolist())  # fft from chest channel
N = len(stomach_data)  # Number of data points
# Plot the fft for the entire time
# visualization.plot_fft(N, time_step, stomach_fft, chest_fft)

# Moving recordings
fft_fig, fft_ax = visualization.initialize_fft_frame()  # Initialize the figure
window_size = 10  # Width of the recording window in seconds
window_step = 1  # Step size in seconds (how far window moves in each iteration)

motors = [None]  # List of indices that the motor starts vibrating

# For i such that we start at 0 and go until the last window reaches the end of the data (about 120 iterations)
for i in range(int(round(time_points[-1] - window_size)/window_step)):

    stomach_sample = stomach_data[i*window_step*sample_rate:(i*window_step+window_size)*sample_rate]  # Get stomach data within window
    chest_sample = chest_data[i*window_step*sample_rate:(i*window_step+window_size)*sample_rate]  # Get chest data within window
    stomach_fft = fft(stomach_sample.tolist())  # Do FFT on stomach sample
    chest_fft = fft(chest_sample.tolist())  # Do FFT on chest sample
    N = len(stomach_sample)  # Number of points within the sample

    # If max power of chest is greater than 10 and there hasn't been a stimulus in the last 10 iterations
    if (max(2.0 / N * np.abs(chest_fft[0:N // 2])) > 10) and ((len(motors) == 1) or (abs(motors[-1]) < i-10)):
        motors = motors + [i]  # Record this as a motor start time

    # Peak analysis in frequency domain, if breathing rate is more than max breaths/minute, triple buzz
    max_bpm = 5
    max_amp_fft = np.argmax(np.abs(stomach_fft))
    power = np.abs(stomach_fft)**2
    x_freq = fftfreq(N, time_step)[:N // 2]  # x frequencies from online example
    max_freq = x_freq[max_amp_fft]
    peak_freq = x_freq[power.argmax()]*2

    # Peak analysis in time domain, if breathing rate is more than max breaths/minute, triple buzz
    # peaks = find_peaks(stomach_sample, distance=None)
    # rate = peaks[0].size/(len(stomach_sample)/200)
    if (peak_freq > max_bpm) and ((len(motors) == 1) or (abs(motors[-1]) < i-10)):
        motors = motors + [-i]

    # Plot the FFTs, time domain data, and motor pulses in the same plot
    visualization.plot_fft_frame(N, stomach_fft, chest_fft, x_freq, fft_ax, stomach_data, chest_data,
                                 time_points, window_step, window_size, i, motors)



