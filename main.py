from modules.preprocessing import *
from modules.maintenence import *
from modules.stream_data import *
from modules.ssvep_handler import *
from modules.stim_pres import *

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from pynput import keyboard

## Adjust As Necessary
serial_port = 'COM7' # Insert port where Cyton Dongle is inserted. This looks different on MAC/Linux -> "/dev/tty*"
board_id = BoardIds.SYNTHETIC_BOARD #BoardIds.CYTON_BOARD # BoardIds.UNICORN_BOARD  # Other Boards: https://brainflow.readthedocs.io/en/stable/UserAPI.html#brainflow-board-shim
# frequencies = [9.25, 11.25, 13.25, 15.25] # Stimulus frequencies; used for CCA & harmonic generation
frequencies = [9.25, 13.25, 17.25, 21.25]
# buttons = ['Right', 'Left', 'Up', 'Down'] # Adds custom text to each box - must be same length as frequencies 
button_pos = [0, 2, 3, 1] # Assigns positions to custom text - must be same length as buttons
segment_duration = 5 # seconds
display = 0 # Which screen to display the stimulus paradigm on --> 0 is default

# Static Variables - Probably don't need to touch :)
harmonics = np.arange(1, 6) # Generates the 1st, 2nd, & 3rd Harmonics
sampling_rate = BoardShim.get_sampling_rate(board_id)
n_samples = sampling_rate * segment_duration 

eeg_channels = BoardShim.get_eeg_names(board_id)
channel_names = ["O1", "O2", "Oz", "Pz", "P3", "P4", "POz", "P1"]
channel_mapping = dict(zip(eeg_channels, channel_names))

# Show board information
print(f"Sampling Rate: {sampling_rate}")
print(f"Default Channels: {eeg_channels}")
print(f"Channel Mapping: {channel_mapping}")

def run_stimulus():
    stimulus = SSVEPStimulus(frequencies, box_text_indices=button_pos, show_both=True, display_index=display) # box_texts=buttons,
    stimulus.run()

def main():
    
    ## Listening for `esc` key to exit
    key_listener = KeyListener()
    key_listener.run_listener()

    ## Initializing Board  
    board = BrainFlowBoardSetup(board_id, serial_port)
    # board.show_params() # Logger shows this info by default - this is another method to show
    board.setup()
    
    # Letting 5 seconds of data accumulate
    print(f"....Warming up....")
    time.sleep(5)
    
    # Reading Data
    # data = board.get_board_data()
    # data = board.get_current_board_data(num_samples=1250)
    # print(f"Collected data shape: {data.shape}")
    # print(f"(Channels, Samples)")
    
    ## Initializing Segmenter Class
    segmenter = PreProcess(board, segment_duration=segment_duration)
    
    # Initialize the SSVEP Classification & Harmonics handler
    classifier = ClassifySSVEP(frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=False)
    classifier_stacked = ClassifySSVEP(frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=True)
    fbcca_classifier = FBCCA(frequencies, harmonics, sampling_rate, n_samples)
    
    # Start the stimulus presentation in a separate thread
    stimulus_thread = threading.Thread(target=run_stimulus)
    stimulus_thread.start()
    
    try:
        while not key_listener.stop_flag:
            # Step 1: Get a segment of data
            segment = segmenter.get_segment()
            if segment is not None:
                
                # print(f"Segment shape: {segment.shape}")

                # Only the first 8 channels are EEG data, so we slice out the remaining channels (9-24)    
                eeg_segment = segment[:8, :]

                # Step 2: Filter the data
                filtered_segment = segmenter.filter_data(eeg_segment)
                print("Filtered data shape:", filtered_segment.shape)

                # Step 3: Use CCA to match the EEG & Reference (harmonic) signals
                    # Unstacked Harmonics (testing)
                detected_freq, correlation = classifier.cca_analysis(filtered_segment)
                print(f"Detected frequency: {detected_freq} Hz with correlation: {correlation}")
                
                detected_freq_stacked, correlation_stacked = classifier_stacked.cca_analysis(filtered_segment)
                print(f"Stacked CCA: Detected frequency: {detected_freq_stacked} Hz with correlation: {correlation_stacked}")

                detected_freq_fbcca, correlation_fbcca = fbcca_classifier.fbcca_analysis(filtered_segment)
                print(f"FBCCA: Detected frequency: {detected_freq_fbcca} Hz with correlation: {correlation_fbcca}")

                # Check SNR for each target frequency
                snr_results = classifier_stacked.check_snr(filtered_segment)
                for freq, snr in snr_results.items():
                    print(f"Frequency: {freq} Hz, SNR: {snr:.2f} dB")

                # Optionally save or process the data further
                # segmenter.save_data(filtered_data, "filtered_data.csv")
                # segmenter.save_data(features, "features.csv")

            # Sleep for a while to collect new data
            time.sleep(segmenter.segment_duration)

    except KeyboardInterrupt:
        pass

    finally:
        board.stop()
        print("\nSession Exited Successfully\n")
        
        
        
if __name__ == "__main__":     
    main()