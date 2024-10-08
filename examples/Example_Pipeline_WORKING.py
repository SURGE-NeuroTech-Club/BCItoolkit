import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import *
from modules.filtering import *
from modules.segmentation import *
from modules.classification import *
from modules.ssvep_stim import *

# Setting variables:
board_id = BoardIds.SYNTHETIC_BOARD.value #BoardIds.CYTON_BOARD.value 
frequencies = [9.25, 11.25, 13.25, 15.25]
buttons = ['Right', 'Left', 'Up', 'Down']
button_pos = [0, 2, 3, 1]
display = 0
segment_duration = 2

# Static Variables - Probably don't need to touch :)
harmonics = np.arange(1, 4) # Generates the 1st, 2nd, & 3rd Harmonics
sampling_rate = BoardShim.get_sampling_rate(board_id)
n_samples = sampling_rate * segment_duration

def main():
    
    # Initialize Streaming Board
    board = BrainFlowBoardSetup(board_id = board_id, serial_port = 'COM3')
    board.setup()

    print(f"Sampling Rate: {sampling_rate}")
    print(f"Number of Samples: {n_samples}")

    # Run the SSVEP Stimulus in a seperate process
    stimulus_process = SSVEPStimulusRunner(box_frequencies = frequencies, 
                                            box_texts = buttons, 
                                            box_text_indices = button_pos,
                                            display_index = display,
                                            display_mode = 'both')

    stimulus_process.start()
    
    # Wait to give SSVEP stimulus time to start and get the actual frequencies
    time.sleep(10)

    actual_freqs = stimulus_process.get_actual_frequencies()
    print("Actual Frequencies:", actual_freqs)

    # Initialize the classifier
    cca_classifier = SSVEPClassifier(frequencies = actual_freqs, 
                                    harmonics = harmonics, 
                                    sampling_rate = sampling_rate, 
                                    n_samples= n_samples, 
                                    method = 'CCA', 
                                    stack_harmonics = True)

    # segmentation_time_wait = Segmentation(board, segment_duration=2)
    # segmenter = Segmentation(board, )

    filter_obj = Filtering(sampling_rate)

    # Start segmentation loop
    while True:
        # segment = segmentation_time_wait.get_segment_time()

        segment = board.get_current_board_data(num_samples = n_samples)

        # print(f"Total shape: {segment.shape}")

        eeg_segment = segment[:8, :]  # Only the first 8 channels are EEG channels
        print(f"Segment Shape: {eeg_segment.shape}")
        
        filtered_segment = filter_obj.bandpass_filter(eeg_segment,
                                                highcut=30,
                                                lowcut=0.1,
                                                order=4  # Parameter adjusts rolloff of filter (higher = faster dropoff)
                                                )
        
        # print(filtered_segment[:, :30])
        
        detected_freq, correlation = cca_classifier(filtered_segment)
        print(f"Detected frequency using CCA: {detected_freq} Hz with correlation: {correlation:.3f}")

        # Wait for 2 seconds to accumulate new data
        time.sleep(segment_duration)


if __name__ == "__main__":
    main()