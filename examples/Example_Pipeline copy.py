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

def run_stimulus(stimulus):
    while not stimulus.is_finished():
        stimulus.update()

def process_eeg(segmentation_time_wait, cca_classifier):
    while True:
        segment = segmentation_time_wait.get_segment_time()
        if segment is not None:
            print("Segment Retrieved:", segment.shape)
            eeg_segment = segment[:8, :]  # Adjust this as needed

            detected_freq, correlation = cca_classifier(eeg_segment)
            print(f"Detected frequency using CCA: {detected_freq} Hz with correlation: {correlation:.3f}")

        time.sleep(0.1)

def main():
    serial_port = 'COM4'
    board_id = BoardIds.SYNTHETIC_BOARD
    frequencies = [9.25, 11.25, 13.25, 15.25]
    buttons = ['Right', 'Left', 'Up', 'Down']
    button_pos = [0, 2, 3, 1]
    segment_duration = 4
    display = 0

    board = BrainFlowBoardSetup(board_id, serial_port)
    board.setup()

    stimulus = SSVEPStimulus(box_frequencies=frequencies, 
                             box_texts=buttons, 
                             box_text_indices=button_pos,          
                             display_mode="both",
                             display_index=display)

    segmentation_time_wait = Segmentation(board, segment_duration=2)
    actual_freqs = stimulus.actual_frequencies

    cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
                                     harmonics=np.arange(1, 4), 
                                     sampling_rate=BoardShim.get_sampling_rate(board_id), 
                                     n_samples=sampling_rate * segment_duration, 
                                     method='CCA', 
                                     stack_harmonics=True)

    stimulus.start()

    stimulus_thread = threading.Thread(target=run_stimulus, args=(stimulus,))
    stimulus_thread.start()

    eeg_thread = threading.Thread(target=process_eeg, args=(segmentation_time_wait, cca_classifier))
    eeg_thread.start()

    stimulus_thread.join()
    eeg_thread.join()

    stimulus.stop()

if __name__ == "__main__":
    main()