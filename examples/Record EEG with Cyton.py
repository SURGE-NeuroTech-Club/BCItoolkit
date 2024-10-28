import sys
import os
import time
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import *
from modules.brainflow_stream import *
from modules.filtering import *
from modules.segmentation import *
from modules.classification import *
from modules.ssvep_stim import *
from brainflow import BoardShim, BrainFlowInputParams, BoardIds

# Setting variables:
board_id = BoardIds.CYTON_BOARD.value  # Use Cyton board
frequencies = [9.25, 11.25, 13.25, 15.25]
buttons = ['Right', 'Left', 'Up', 'Down']
button_pos = [0, 2, 3, 1]
display = 0
segment_duration = 10  # in seconds

# Static Variables
sampling_rate = BoardShim.get_sampling_rate(board_id)

def record_long_format_data(board, filename='ssvep_long_format.csv', duration=120):
    """
    Records long-format continuous data from the OpenBCI board and saves it to a CSV file.
    """
    print("Starting data stream...")
    board.start_stream()  # Start streaming data
    
    # Wait for the specified duration to collect enough data
    time.sleep(duration)
    
    # Stop the data stream
    data = board.get_board_data()  # Retrieve all data collected so far
    print(f"Collected {data.shape[1]} samples.")

    # Convert the data to a pandas DataFrame for easier saving
    df = pd.DataFrame(data.T)  # Transpose to ensure each row is a sample
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False, header=False)
    print(f"Data saved to {filename}.")

def main():
    # Initialize BrainFlow input params and the board
    params = BrainFlowInputParams()
    params.serial_port = 'COM4'  # Specify the correct COM port
    
    board = BoardShim(board_id, params)
    board.prepare_session()

    # Run the SSVEP Stimulus in a separate process
    stimulus_process = SSVEPStimulusRunner(box_frequencies=frequencies,
                                            box_texts=buttons,
                                            box_text_indices=button_pos,
                                            display_index=display,
                                            display_mode='both')
    stimulus_process.start()

    actual_freqs = stimulus_process.get_actual_frequencies()
    print("Actual Frequencies:", actual_freqs)
    
    # Wait for the SSVEP stimulus to stabilize
    time.sleep(10)

    # Start recording the long-format data
    record_long_format_data(board, filename='120s_cyton_recording.csv')

    # Stop the stimulus process and release board session
    stimulus_process.stop()
    board.release_session()



if __name__ == "__main__":
    main()
