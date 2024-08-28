import sys
sys.path.append("..")
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import numpy as np
from mvlearn.embed import CCA
import matplotlib.pyplot as plt

# Library Modules:
from modules.brainflow_stream import *
from modules.filtering import *
from modules.segmentation import *
from modules.psychopy_ssvep_stim import *

from modules.classification import *


def main():
    serial_port = 'COM4' 
    board_id = BoardIds.SYNTHETIC_BOARD # BoardIds.PLAYBACK_FILE_BOARD -> only if recorded with a support board (I believe...)
    frequencies = [9.25, 11.25, 13.25, 15.25] # Stimulus frequencies; used for CCA & harmonic generation
    buttons = ['Right', 'Left', 'Up', 'Down'] # Adds custom text to each box - must be same length as frequencies 
    button_pos = [0, 2, 3, 1] # Assigns positions to custom text - must be same length as buttons
    segment_duration = 4 # seconds
    display = 0 # Which screen to display the stimulus paradigm on --> 0 is default

    # Static Variables - Probably don't need to touch :)
    harmonics = np.arange(1, 4) # Generates the 1st, 2nd, & 3rd Harmonics for CCA/FBCCA/foCCA
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    n_samples = sampling_rate * segment_duration 


    eeg_channels = BoardShim.get_eeg_names(board_id)
    channel_names = ["O1", "O2", "Oz", "Pz", "P3", "P4", "POz", "P1"]
    channel_mapping = dict(zip(eeg_channels, channel_names))

    # Show board information
    print(f"Sampling Rate: {sampling_rate}")
    print(f"Default Channels: {eeg_channels}")
    print(f"Channel Mapping: {channel_mapping}")

    board = BrainFlowBoardSetup(board_id, serial_port)
    # board.show_params() # Logger shows this info by default - this is another method to show
    board.setup()


    stimulus = SSVEPStimulus(box_frequencies=frequencies, 
                            box_texts=buttons, 
                            box_text_indices=button_pos,          
                            display_mode="text",
                            display_index=display)


if __name__ == "__main__":
    main()