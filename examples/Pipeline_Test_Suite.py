import sys
import os
import time
import matplotlib.pyplot as plt
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
segment_duration = 4

# Static Variables - Probably don't need to touch :)
harmonics = np.arange(1, 4) # Generates the 1st, 2nd, & 3rd Harmonics
sampling_rate = BoardShim.get_sampling_rate(board_id)
n_samples = sampling_rate * segment_duration


def visualize_eeg_signals(eeg_segment, filtered_segment):
    """
    Visualizes the raw and filtered EEG signals for comparison.

    Args:
        eeg_segment (np.ndarray): The raw EEG data (n_channels, n_samples).
        filtered_segment (np.ndarray): The filtered EEG data (n_channels, n_samples).
    """
    n_channels = eeg_segment.shape[0]
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw EEG signals
    plt.subplot(2, 1, 1)
    for i in range(n_channels):
        plt.plot(eeg_segment[i, :], label=f'Channel {i+1}')
    plt.title("Raw EEG Signals")
    plt.legend()
    
    # Plot filtered EEG signals
    plt.subplot(2, 1, 2)
    for i in range(n_channels):
        plt.plot(filtered_segment[i, :], label=f'Filtered Channel {i+1}')
    plt.title("Filtered EEG Signals")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_reference_vs_eeg(eeg_segment, reference_signal, frequency):
    """
    Visualizes a reference signal against the EEG data for comparison.

    Args:
        eeg_segment (np.ndarray): The raw or filtered EEG data (n_channels, n_samples).
        reference_signal (np.ndarray): The reference signal (n_samples,).
        frequency (float): The frequency of the reference signal.
    """
    plt.figure(figsize=(10, 4))

    # Plot one of the EEG channels (assuming EEG data is (n_channels, n_samples))
    plt.plot(eeg_segment[0, :], label="EEG Channel 1", color="blue")
    
    # Plot the reference signal
    plt.plot(reference_signal[:, 0], label=f"Reference Signal ({frequency} Hz)", color="red")
    
    plt.title(f"Comparison: EEG Signal vs Reference Signal ({frequency} Hz)")
    plt.legend()
    plt.show()

def plot_correlation_across_frequencies(frequencies, correlations):
    """
    Plots the canonical correlation values for each frequency.

    Args:
        frequencies (list): List of target frequencies.
        correlations (list): Corresponding correlation values.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(frequencies, correlations, color='skyblue')
    plt.title("Canonical Correlation Across Frequencies")
    plt.xlabel("Frequencies (Hz)")
    plt.ylabel("Correlation Coefficient")
    plt.show()


def check_signal_alignment(eeg_segment, reference_signal):
    """
    Checks whether the EEG segment and reference signals are aligned in terms of length.

    Args:
        eeg_segment (np.ndarray): The EEG data (n_channels, n_samples).
        reference_signal (np.ndarray): The reference signals (n_samples, n_features).
    """
    eeg_len = eeg_segment.shape[1]
    ref_len = reference_signal.shape[0]
    
    if eeg_len != ref_len:
        print(f"Warning: Mismatch in lengths. EEG samples: {eeg_len}, Reference samples: {ref_len}")
    else:
        print(f"EEG and reference signals are aligned (Length: {eeg_len}).")

def main():
    # Initialize Streaming Board
    board = BrainFlowBoardSetup(board_id=board_id, serial_port='COM3')
    board.setup()

    # Run the SSVEP Stimulus in a separate process
    stimulus_process = SSVEPStimulusRunner(box_frequencies=frequencies, 
                                           box_texts=buttons, 
                                           box_text_indices=button_pos,
                                           display_index=display,
                                           display_mode='both')
    # stimulus_process.start()

    # Wait for the SSVEP stimulus to stabilize
    time.sleep(10)

    # actual_freqs = stimulus_process.get_actual_frequencies()
    actual_freqs = frequencies
    print("Actual Frequencies:", actual_freqs)

    cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
                                     harmonics=harmonics, 
                                     sampling_rate=sampling_rate, 
                                     n_samples=n_samples, 
                                     method='CCA', 
                                     stack_harmonics=True)

    filter_obj = Filtering(sampling_rate)

    while True:
        segment = board.get_current_board_data(num_samples=n_samples)
        eeg_segment = segment[:8, :]  # First 8 channels assumed to be EEG
        filtered_segment = filter_obj.bandpass_filter(eeg_segment, highcut=30, lowcut=0.1, order=4)
        
        # Visualize raw vs filtered EEG signals
        visualize_eeg_signals(eeg_segment, filtered_segment)

        # Check signal alignment
        reference_signal = cca_classifier.reference_signals[actual_freqs[1]]
        check_signal_alignment(filtered_segment, reference_signal)

        # Compare EEG vs reference signal
        visualize_reference_vs_eeg(filtered_segment, reference_signal, actual_freqs[0])

        # Perform classification
        correlations = []
        for freq in actual_freqs:
            detected_freq, correlation = cca_classifier(filtered_segment)
            correlations.append(correlation)
        
        # Plot correlation coefficients across frequencies
        plot_correlation_across_frequencies(actual_freqs, correlations)

        # Wait for the next segment
        time.sleep(2)


if __name__ == '__main__':
    main()