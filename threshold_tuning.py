import pandas as pd
import visualization
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfilt
import numpy as np
from scipy import stats

# List of CSVs to be read
files = ["breathing_filtered.csv",
         "JacquieData1_20211126.csv",
         "JacquieData2_20211126.csv",
         "JacquieDec1_1.csv",
         "JacquieDec1_2.csv",
         "RayDec1_1.csv",
         "RayDec1_2.csv"]

# Threshold ranges for breath per minute (bpm) and chest power
chest_thresh = range(1, 20)
bpm_thresh = range(1, 20)

# Moving recordings
window_size = 5  # Width of the recording window in seconds
window_step = 1  # Step size in seconds (how far window moves in each iteration)

chest_true_pos = [None for i in chest_thresh]  # Sensitivity
chest_false_pos = [None for i in chest_thresh]  # Specificity
accuracies = [None for i in chest_thresh]  # Accuracy

for t, ct in enumerate(chest_thresh):
    positives = []  # Container for all true positive (1) and false positive (0) decisions
    negatives = []  # Container for all true negative (1) and false negative (0) decisions
    print("THRESHOLD: ", ct)

    for fn, f in enumerate(files):
        # print("File %d/%d" % (fn + 1, len(files)))
        # Load in the csv
        breathing_filtered = pd.read_csv(f)
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
        # Labels: 0 (regular), 1 (chest), 2 (diaphragm), 3 (fast breathing), 4 (low freq motion)
        events = breathing_filtered_sparse["Event"]
        event_times = visualization.get_event_times(events, time_points)

        # Plot the data in the time domain for both channels
        # visualization.plot_csv(stomach_data, chest_data, time_points, event_times)

        # For i such that we start at 0 and go until the last window reaches the end of the data (about 120 iterations)
        for i in range(int(round(time_points[-1] - window_size)/window_step)):

            stomach_sample = stomach_data[i*window_step*sample_rate:(i*window_step+window_size)*sample_rate]  # Get stomach data within window
            chest_sample = chest_data[i*window_step*sample_rate:(i*window_step+window_size)*sample_rate]  # Get chest data within window
            events_sample = events[i*window_step*sample_rate:(i*window_step+window_size)*sample_rate]  # Get events in window

            # Perform fast fourier transforms
            stomach_fft = fft(stomach_sample.tolist())  # Do FFT on stomach sample
            chest_fft = fft(chest_sample.tolist())  # Do FFT on chest sample
            N = len(stomach_sample)  # Number of points within the sample

            # If max power of chest is greater than threshold, label this window as bad for chest amplitude
            if max(2.0 / N * np.abs(chest_fft[0:N // 2])) > ct:
                chest_label = 1
            else:
                chest_label = 0

            # Get the actual label from the events channel
            this_event = stats.mode(events_sample)[0][0]
            if this_event == 1:  # Chest breathing, incorrect
                this_label = 1
            else:
                this_label = 0

            # Determine if this is true pos, true neg, false pos, false neg
            if chest_label == 1:  # Threshold returned positive
                if this_label == 1:  # True positive
                    positives = positives + [1]
                else:
                    positives = positives + [0]
            else:  # Threshold returned negative
                if this_label == 0:  # True negative
                    negatives = negatives + [1]
                else:
                    negatives = negatives + [0]

    # Get the true positive and true negative rates for this threshold
    true_positives = float(sum(positives))
    false_positives = float(len(positives) - true_positives)
    true_negatives = float(sum(negatives))
    false_negatives = float(len(negatives) - true_negatives)

    chest_true_pos[t] = true_positives / (true_positives + false_negatives)
    chest_false_pos[t] = false_positives / (false_positives + true_negatives)

    accuracies[t] = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)

# Plot the ROC curve
visualization.plot_roc(chest_true_pos, chest_false_pos)

# Calculate the AUC
AUC = 0.0
for case in range(0, len(chest_false_pos) - 1):
    delta_x = chest_false_pos[case] - chest_false_pos[case+1]
    delta_y = chest_true_pos[case]
    AUC = AUC + delta_y*delta_x

print(AUC)
