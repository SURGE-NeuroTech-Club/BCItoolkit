import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import time
import numpy as np
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from mvlearn.embed import CCA

# Library Modules:
from modules.brainflow_stream import *
from modules.filtering import *
from modules.segmentation import *
from modules.psychopy_ssvep_stim import *
from modules.classification import *

# Setup parameters
serial_port = 'COM4'
board_id = BoardIds.SYNTHETIC_BOARD
frequencies = [9.25, 11.25, 13.25, 15.25]
buttons = ['Right', 'Left', 'Up', 'Down']
button_pos = [0, 2, 3, 1]
segment_duration = 4
display = 0

# Static Variables
harmonics = np.arange(1, 4)
sampling_rate = BoardShim.get_sampling_rate(board_id)
n_samples = sampling_rate * segment_duration 

eeg_channels = BoardShim.get_eeg_names(board_id)
channel_names = ["O1", "O2", "Oz", "Pz", "P3", "P4", "POz", "P1"]
channel_mapping = dict(zip(eeg_channels, channel_names))

# Show board information
print(f"Sampling Rate: {sampling_rate}")
print(f"Default Channels: {eeg_channels}")
print(f"Channel Mapping: {channel_mapping}")

def run_stimulus(stimulus):
    stimulus.start()
    while not stimulus.is_finished():
        stimulus.update()
        time.sleep(1.0 / stimulus.refresh_rate)  # Maintain the refresh rate consistency

def run_processing(board, stimulus, harmonics, sampling_rate, n_samples):
    segmentation_time_wait = Segmentation(board, segment_duration=2)
    actual_freqs = stimulus.actual_frequencies

    cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
                                     harmonics=harmonics, 
                                     sampling_rate=sampling_rate, 
                                     n_samples=n_samples, 
                                     method='CCA', 
                                     stack_harmonics=True)

    while not stimulus.is_finished():
        segment = segmentation_time_wait.get_segment_time()
        if segment is not None:
            print("Segment Retrieved:", segment.shape)
            eeg_segment = segment[:8, :] 

            detected_freq, correlation = cca_classifier(eeg_segment)
            print(f"Detected frequency using CCA: {detected_freq} Hz with correlation: {correlation:.3f}")

def main():
    board = BrainFlowBoardSetup(board_id, serial_port)
    board.setup()

    # Initialize stimulus
    stimulus = SSVEPStimulus(box_frequencies=frequencies, 
                             box_texts=buttons, 
                             box_text_indices=button_pos,          
                             display_index=display)

    # Start EEG data processing in a separate thread
    processing_thread = threading.Thread(target=run_processing, args=(board, stimulus, harmonics, sampling_rate, n_samples))
    processing_thread.start()

    # Run stimulus presentation in the main thread
    run_stimulus(stimulus)

    # Ensure the processing thread finishes before exiting
    processing_thread.join()

if __name__ == "__main__":
    main()
