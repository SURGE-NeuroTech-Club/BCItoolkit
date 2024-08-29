import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import *
from modules.filtering import *
from modules.segmentation import *
from modules.classification import *
from modules.multiprocessing_psychopy_ssvep_stim import *

# Setting variables:
serial_port = 'COM4'
board_id = BoardIds.SYNTHETIC_BOARD
frequencies = [9.25, 11.25, 13.25, 15.25]
buttons = ['Right', 'Left', 'Up', 'Down']
button_pos = [0, 2, 3, 1]
display = 0
segment_duration = 4

# Static Variables - Probably don't need to touch :)
harmonics = np.arange(1, 4) # Generates the 1st, 2nd, & 3rd Harmonics
sampling_rate = BoardShim.get_sampling_rate(board_id)
n_samples = sampling_rate * segment_duration 


def main():
    
    # Initialize Streaming Board
    board = BrainFlowBoardSetup(board_id, serial_port)
    board.setup()

    segmentation_time_wait = Segmentation(board, segment_duration=2)
    actual_freqs = frequencies

    # Initialize the classifier
    cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
                                    harmonics=np.arange(1, 4), 
                                    sampling_rate=BoardShim.get_sampling_rate(board_id), 
                                    n_samples=BoardShim.get_sampling_rate(board_id) * segment_duration, 
                                    method='CCA', 
                                    stack_harmonics=True)

    # Run the SSVEP Stimulus in a seperate process
    run_ssvep_stimulus_in_process(box_frequencies=frequencies, 
                                box_texts=buttons, 
                                box_text_indices=button_pos,
                                display_index=display, 
                                display_mode=display_mode)

    # Implement small wait to give SSVEP stimulus time to start
    time.wait(10)

    # Start segmentation loop
    while True:
        segment = segmentation_time_wait.get_segment_time()
        if segment is None:
            time.sleep(0.1)  # Wait a little before trying again
            continue

        eeg_segment = segment[:8, :]  # Assuming 8 channels, adjust if necessary
        detected_freq, correlation = cca_classifier(eeg_segment)
        print(f"Detected frequency using CCA: {detected_freq} Hz with correlation: {correlation:.3f}")

        time.sleep(0.1)


if __name__ == "__main__":
    main()