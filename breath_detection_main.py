import pandas as pd
import visualization
from scipy.fft import fft, fftfreq
import numpy as np

# Read the CSV
breathing_filtered = pd.read_csv("breathing_filtered.csv")
# The original data is sampled at 2 kHz, this is way more than we need, so take only every 200 points (10 Hz)
breathing_filtered_sparse = breathing_filtered[0::200]

# Overall breathing data plot
stomach_data = breathing_filtered_sparse["Stomach"]  # Get stomach data using "Stomach" key
chest_data = breathing_filtered_sparse["Chest"]  # Get chest data using "Chest" key
time_points = breathing_filtered_sparse["Elapsed Time"]  # Get the time points from "Elapsed Time" key
time_step = 1.0/1000  # Time between each data point
max_time = len(time_points)/time_step  # Max time of the entire recording
# Plot the data in the time domain for both channels
visualization.plot_csv(stomach_data, chest_data)

# Overall fft plot
stomach_fft = fft(stomach_data.tolist())  # fft from stomach channel
chest_fft = fft(chest_data.tolist())  # fft from chest channel
N = len(stomach_data)  # Number of data points
# Plot the fft for the entire time
visualization.plot_fft(N, time_step, stomach_fft, chest_fft)

# Moving recordings
window_size = int(len(stomach_data)/10)  # Width of the recording window
window_step = int(len(stomach_data)/100)  # Step size (how far window moves in each iteration)
fft_fig, fft_ax = visualization.initialize_fft_frame()  # Initialize the figure

motors = [None]  # List of indices that the motor starts vibrating

# For i such that we start at 0 and go until the last window reaches the end of the data (about 90 iterations)
for i in range(int(len(stomach_data)/window_step)-10):

    print(i)

    stomach_sample = stomach_data[i*window_step:i*window_step+window_size]  # Get stomach data within window
    chest_sample = chest_data[i*window_step:i*window_step+window_size]  # Get chest data within window
    stomach_fft = fft(stomach_sample.tolist())  # Do FFT on stomach sample
    chest_fft = fft(chest_sample.tolist())  # Do FFT on chest sample
    N = len(stomach_sample)  # Number of points within the sample

    # If max power of chest is greater than 10 and there hasnt been a stimulus in the last 10 iterations
    if (max(2.0 / N * np.abs(chest_fft[0:N // 2])) > 10) and ((len(motors) == 1) or (motors[-1] < i-10)):
        motors = motors + [i]  # Record this as a motor start time

    # Plot the FFTs, time domain data, and motor pulses in the same plot
    visualization.plot_fft_frame(N, time_step, stomach_fft, chest_fft, fft_fig, fft_ax, stomach_data, chest_data,
                                 window_step, window_size, i, motors)



