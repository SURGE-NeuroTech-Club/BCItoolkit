import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
from psychopy import visual, event, core, monitors
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from mvlearn.embed import CCA
from multiprocessing import Process, Queue

# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import BrainFlowBoardSetup
from modules.filtering import *
from modules.segmentation import Segmentation
from modules.classification import SSVEPClassifier
from modules.psychopy_ssvep_stim import *

def start_visual_stimulus(box_frequencies, box_texts, box_text_indices, display_mode, display_index):
    stimulus = SSVEPStimulus(box_frequencies, box_texts, box_text_indices, display_mode, display_index)
    stimulus.start()

def run_eeg_processing(serial_port, board_id, frequencies, buttons, button_pos, segment_duration):
    board = BrainFlowBoardSetup(board_id, serial_port)
    board.setup()

    segmentation_time_wait = Segmentation(board, segment_duration=2)
    actual_freqs = frequencies

    cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
                                     harmonics=np.arange(1, 4), 
                                     sampling_rate=BoardShim.get_sampling_rate(board_id), 
                                     n_samples=BoardShim.get_sampling_rate(board_id) * segment_duration, 
                                     method='CCA', 
                                     stack_harmonics=True)

    while True:
        segment = segmentation_time_wait.get_segment_time()
        if segment is None:
            time.sleep(0.1)  # Wait a little before trying again
            continue

        eeg_segment = segment[:8, :]  # Assuming 8 channels, adjust if necessary
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

    # Start the visual stimulus in a separate process
    stimulus_process = Process(target=start_visual_stimulus, args=(frequencies, buttons, button_pos, display, 'both'))
    stimulus_process.start()

    # Start the EEG processing in a separate process
    eeg_process = Process(target=run_eeg_processing, args=(serial_port, board_id, frequencies, buttons, button_pos, segment_duration))
    eeg_process.start()

    # Wait for both processes to complete
    stimulus_process.join()
    eeg_process.join()

if __name__ == "__main__":
    main()