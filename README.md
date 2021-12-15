# BioFB

Using this repo for capstone data/code. Please put things in folders & keep organized!

INFO on data:
- ALL data is sampled at a rate of 2 kHz
- Matlab workspaces and code were for proof of concept work with the neural nets and are not implemented in final code
- .csv files in LabeledBreathingData folder were manually labelled with behaviour (chest breathing, diahragmatic breathing, etc.), labels are outlined in breathdetectionmain.py 
- BreathingData folder contains all raw data collected from BioRadio, LabeledBreathingData contains .csv's that were used in the code

Code descriptions
- bpm_threshold_tuning.py: tests different thresholds for maximum breaths per minute to determine highest accuracy
- breath_detection_main.py: contains the code to run the demo
- create_dataset.py: compiles sample .csv's in LabeledBreathingData folder and shuffles them to create a training/testing set for the neural net
- data_processing_pseudocode.py: pseudocode from the report
- hexotest.py: code for testing the hexoskin, not used in final prototype
- learning.py: contains code for training and testing the neural network
- network_testing.m: contains code for proof of concept testing of neural networks in matlab
- pythonserial.py: communication with arduino test
- sklearn_nn.joblib: this is the neural network that we trained and used
- threshold_tuning.py: code for testing different chest thresholds
- visualization.py: contains helper functions for making plots 
