from matplotlib import pyplot as plt
from scipy.fft import fftfreq
import numpy as np
from matplotlib.patches import Rectangle


def get_event_times(event_array, time_array):
    event_times = []
    event_list = event_array.tolist()
    for i in range(1, len(event_list)):
        if (event_list[i] - event_list[i-1]) != 0:
            event_times = event_times + [time_array[i]]
    return event_times


def plot_csv(stomach, chest, time_array, vlines):
    # Plot the time domain data from both channels
    # Create plot with 2 subplots sharing the same x axis
    fig, ax = plt.subplots(2, sharex=True, figsize=(19, 9))

    # On top plot, plot the chest
    ax[0].plot(time_array, chest)
    ax[0].set_title("Breathing Sample (BioRadio)")
    ax[0].set_ylabel("Chest")

    # On bottom plot, plot the stomach
    ax[1].plot(time_array, stomach)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Stomach")

    # Vertical lines delineating different behaviour regions
    if vlines:
        for i in vlines:
            ax[0].axvline(i, color='k', ls='--')
            ax[1].axvline(i, color='k', ls='--')

    plt.xlim([0, time_array[-1]])  # Set xlimits to be entire recording
    plt.show()  # Display the plot


def plot_fft(N, T, stomach_fft, chest_fft):
    # Plot the FFT for each channel over the entire recording
    x_freq = fftfreq(N, T)[:N // 2]  # Frequencies along the x axis (taken from online example)
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=[19, 9])  # Create plot with 2 subplots sharing same y-axis
    ax[0].plot(x_freq, 2.0 / N * np.abs(chest_fft[0:N // 2]))  # Plot amplitude of chest FFT on first left plot
    ax[0].set_xlabel("Chest")
    ax[0].grid()  # Show the grid
    ax[0].set_xlim([0, 30])

    ax[1].plot(x_freq, 2.0 / N * np.abs(stomach_fft[0:N // 2]))  # Plot amplitude of stomach FFT on first left plot
    ax[1].set_xlabel("Stomach")
    ax[1].grid()  # Show the grid
    ax[1].set_xlim([0, 30])

    # Overall titles and labels for the entire figure
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Amplitude")
    fig.suptitle("Fast Fourier Transform")
    plt.show()  # Display the plot


def initialize_fft_frame():
    # Set up the FFT per frame plot (done in a separate function so each frame doesn't create new figure)
    fig = plt.figure(figsize=[19, 9])
    ax = [None, None, None, None, None, None]  # List for all the axis objects
    ax[0] = fig.add_subplot(6, 2, 1)  # Add subplot in first index of a 4 row x 2 column grid (for chest fft)
    ax[0].set_ylim([0, 20])  # Set ylimit 20
    ax[1] = fig.add_subplot(5, 2, 2, sharey=ax[0]) # Add subplot in second index of a 4 row x 2 column grid (for stomach fft)
    ax[2] = fig.add_subplot(5, 2, (3, 4))  # Add subplot in 3rd-4th index of a 4 row x 2 column grid (for chest time data)
    ax[3] = fig.add_subplot(5, 2, (5, 6), sharex=ax[2])  # Add subplot in 5th-6th index of a 4 row x 2 column grid (for stomach time data)
    ax[4] = fig.add_subplot(5, 2, (7, 8), sharex=ax[2])  # Add subplot in 7th-8th index of a 4 row x 2 column grid (for motor pulses)
    ax[5] = fig.add_subplot(5, 2, (9, 10), sharex=ax[2])  # Add subplot for neural network label
    # ax[6] = fig.add_subplot(6, 2, (11, 12), sharex=ax[2])  # Add subplot for neural network label
    fig.suptitle("Fast Fourier Transform Over Time")
    fig.supxlabel("Time (s)")
    return fig, ax


def plot_fft_frame(N, stomach_fft, chest_fft, freqs, ax, stomach, chest, time_array, vlines,
                   window_step, window_size, i, motors, nn_outputs):

    # Plot magnitude of chest and stomach FFTs on zeroth and first axis objects
    ax[0].plot(freqs, 2.0 / N * np.abs(chest_fft[0:N // 2]), 'b')
    ax[1].plot(freqs, 2.0 / N * np.abs(stomach_fft[0:N // 2]), 'b')
    ax[0].grid()
    ax[0].set_ylim([0, 20])
    ax[0].set_xlim([0, 30])
    ax[0].set_title("Chest")
    ax[1].grid()
    ax[1].set_title("Stomach")
    ax[1].set_xlim([0, 30])

    # Plot time domain data from two data channels on 2nd and 3rd axis objects
    ax[2].plot(time_array, chest)
    ax[2].set_ylabel("Chest")
    ax[2].axvline(i*window_step, color='g', ls='--')  # Vertical line for beginning of window
    ax[2].axvline(i*window_step+window_size, color='g', ls='--')  # Vertical line for end of window
    ax[2].set_xlim([0, time_array[-1]])
    ax[3].plot(time_array, stomach)
    ax[3].set_ylabel("Stomach")
    ax[3].axvline(i*window_step, color='g', ls='--')  # Vertical line for beginning of window
    ax[3].axvline(i*window_step+window_size, color='g', ls='--')  # Vertical line for end of window

    # Vertical lines delineating different behaviour regions
    if vlines:
        for i in vlines:
            ax[2].axvline(i, color='k', ls='--')
            ax[3].axvline(i, color='k', ls='--')

    # Add motor pulses as rectangles on the last axis object
    ax[4].set_ylim([0, 1])  # Motor is only on (1) or off (0)
    ax[4].set_ylabel("Motor")
    for j in motors[1:]:  # For all of the times that a motor pulse starts
        if j < 0:  # Signal is that breathing rate is too high, do three short pulses
            j = -j
            # Add three red rectangles of width 1000
            ax[4].add_patch(Rectangle((j*window_step+window_size, 0),  # Lower left point of rectangle
                               0.5, 1,  # Width, height of rectangle
                               fc='r',  # Fill colour
                               ec='r',  # Edge colour
                               lw=1))  # Line width
            ax[4].add_patch(Rectangle((j*window_step+window_size + 2, 0),  # Lower left point of rectangle
                               0.5, 1,  # Width, height of rectangle
                               fc='r',  # Fill colour
                               ec='r',  # Edge colour
                               lw=1))  # Line width
            ax[4].add_patch(Rectangle((j*window_step+window_size + 4, 0),  # Lower left point of rectangle
                               0.5, 1,  # Width, height of rectangle
                               fc='r',  # Fill colour
                               ec='r',  # Edge colour
                               lw=1))  # Line width

        else:  # Signal is that chest breathing is too much, do one pulse
            ax[4].add_patch(Rectangle((j*window_step+window_size, 0),  # Lower left point of rectangle
                                   2, 1,  # Width, height of rectangle
                                   fc='k',  # Fill colour
                                   ec='k',  # Edge colour
                                   lw=1))  # Line width

    # Add neural network-produced label
    ax[5].set_ylabel("Neural Net Output")
    width = window_step
    start = 0
    for nn in range(len(nn_outputs)):
        if (nn != len(nn_outputs)-1) and (nn_outputs[nn][0] == nn_outputs[nn+1][0]):
            width = width + window_step  # If this output is same as next, increase width so fewer shapes are drawn
        else:  # Output not same as next or this is the last output, draw the shape
            if nn_outputs[nn][0] == 1:
                ax[5].add_patch(Rectangle((start * window_step + window_size, 0),  # Lower left point of rectangle
                                width, 1,  # Width, height of rectangle
                                fc='b',  # Fill colour
                                ec='b',  # Edge colour
                                lw=1))  # Line width
            else:
                ax[5].add_patch(Rectangle((start * window_step + window_size, 0),  # Lower left point of rectangle
                                          width, 1,  # Width, height of rectangle
                                          fc='r',  # Fill colour
                                          ec='r',  # Edge colour
                                          lw=1))  # Line width
            start = start + width  # Start next rectangle directly after this one
            width = window_step  # Reset the width of the rectangle

    plt.draw()  # Draw all of the plots onto the figure
    plt.pause(0.01)  # Wait for a tenth of a second
    # Clear all of the axes (to make room for the next timepoint and make it look animated)
    for axis in ax:
        axis.cla()


def plot_roc(sensitivity, specificity):
    plt.figure(figsize=[19, 9])
    plt.title("ROC Curve for Chest Thresholds")
    plt.xlabel("False Positive Rate")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    for case in range(len(sensitivity) - 1):
        plt.axvline(x=specificity[case + 1], ymin=sensitivity[case], ymax=sensitivity[case+1])  # Vertical line at the specificity point
        plt.axhline(y=sensitivity[case], xmin=specificity[case], xmax=specificity[case+1])  # Horizontal line at the sensitivity point
    plt.axvline(specificity[4], ls='--', color='k')
    plt.axvline(specificity[6], ls='--', color='k')
    plt.axvline(specificity[9], ls='--', color='k')
    plt.show()

