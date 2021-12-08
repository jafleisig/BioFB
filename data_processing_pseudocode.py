# Data processing module pseudocode

# Load the data from .csv file
data = read_csv("Filename")

# Get the stomach and chest channels
stomach_data = data["Stomach"]
chest_data = data["Chest"]

# Filter with 20 Hz lowpass butterworth filter
filter = butterworth(type = lowpass, threshold = 20)  # Create the filter
stomach_data = apply_filter(stomach_data, filter)  # Apply to stomach data
chest_data = apply_filter(chest_data, filter)  # Apply to chest data

# Go over data in 5 second time windows, with a step size of 1 second
for each time_window:
   
    # Get the sample from each channel in the window
    stomach_sample = stomach_data[time_window]
    chest_sample = chest_data[time_window]
    
    # Get the fast fourier transform (fft's)
    stomach_fft = fft(stomach_sample)
    chest_fft = fft(chest_sample)
    
    # Input the fft's into the neural network
    label = neural_network(stomach_fft, chest_fft)
    
    # If the breathing is labelled as bad, check why using hardcoded methods
    if label is good_breathing:
        motor_pulse = None
        
    else:
        if chest_amplitude > chest_amplitude_threshold:
            motor_pulse = one_long_pulse
        
        elseif breaths_per_minute > breaths_per_minute_threshold:
            motor_pulse = three_short_pulse
        
        else:
            motor_pulse = None
            
    # Write the motor pulse to the arduino to drive the motor
    if motor_pulse is not None:
        write_to_arduino(motor_pulse)
    