import sys
import os
import time
import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, lfilter
from scipy import signal
import neurokit2 as nk
import plotly.graph_objs as go
from plotly.subplots import make_subplots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import *
from modules.filtering import *
from modules.segmentation import *
from modules.classification import *
from modules.ssvep_stim import *

# Setting variables:
board_id = BoardIds.CYTON_BOARD.value # BoardIds.SYNTHETIC_BOARD.value 
frequencies = [9.25, 11.25, 13.25, 15.25]
buttons = ['Right', 'Left', 'Up', 'Down']
button_pos = [0, 2, 3, 1]
display = 0
segment_duration = 10

# Static Variables - Probably don't need to touch :)
harmonics = np.arange(1, 4) # Generates the 1st, 2nd, & 3rd Harmonics
sampling_rate = BoardShim.get_sampling_rate(board_id)
n_samples = sampling_rate * segment_duration

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a bandpass filter to the EEG data.

    Args:
        data (np.ndarray): The input EEG data (n_channels, n_samples).
        lowcut (float): The lower frequency bound of the bandpass filter.
        highcut (float): The upper frequency bound of the bandpass filter.
        fs (float): The sampling rate of the EEG data.
        order (int): The order of the Butterworth filter.

    Returns:
        np.ndarray: The bandpass-filtered EEG data.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to each channel
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

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
    plt.plot(eeg_segment[1, :], label="EEG Channel 1", color="blue")
    
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

def check_dc_offset(eeg_segment):
    """
    Prints the mean value of the EEG data to check for DC offset.
    
    Args:
        eeg_segment (np.ndarray): The raw EEG data (n_channels, n_samples).
    """
    for i in range(eeg_segment.shape[0]):
        print(f"Mean of channel {i+1}: {np.mean(eeg_segment[i, :])}")

def remove_dc_offset(eeg_data):
    """
    Removes the DC offset (mean) from each channel in the EEG data for visualization.
    
    Args:
        eeg_data (np.ndarray): The raw EEG data (n_channels, n_samples).
    
    Returns:
        np.ndarray: The EEG data with the DC offset removed.
    """
    # return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    hp_cutoff_Hz = 1.0
    
    b, a = signal.butter(2, hp_cutoff_Hz / (sampling_rate / 2.0), 'highpass')
    filtered_data = signal.lfilter(b, a, eeg_data, 0)
    # filtered_data = signal.filtfilt(b, a, eeg_data, axis=1)
    
    return filtered_data
    

def visualize_all_channels_plotly(eeg_segment, filtered_segment):
    """
    Visualizes all EEG channels side-by-side using Plotly for interactive viewing.
    
    Args:
        eeg_segment (np.ndarray): The raw EEG data (n_channels, n_samples).
        filtered_segment (np.ndarray): The filtered EEG data (n_channels, n_samples).
    """
    n_channels = eeg_segment.shape[0]
    
    # Create subplots with n_channels rows and 2 columns
    fig = make_subplots(rows=n_channels, cols=2, shared_xaxes=True,
                        subplot_titles=[f'Raw Channel {i+1}' for i in range(n_channels)] +
                                       [f'Filtered Channel {i+1}' for i in range(n_channels)],
                        vertical_spacing=0.05)
    
    # Add traces for each channel (raw and filtered)
    for i in range(n_channels):
        # Raw signal
        fig.add_trace(go.Scatter(y=eeg_segment[i, :], mode='lines', name=f'Raw Channel {i+1}', line=dict(color='blue')),
                      row=i+1, col=1)
        
        # Filtered signal
        fig.add_trace(go.Scatter(y=filtered_segment[i, :], mode='lines', name=f'Filtered Channel {i+1}', line=dict(color='green')),
                      row=i+1, col=2)
    
    # Update layout for better spacing
    fig.update_layout(height=300 * n_channels, width=1200, showlegend=False, title_text="EEG Data: Raw and Filtered")
    
    # Show the figure
    fig.show()

def subtract_reference(eeg_segment, reference_channel):
    """
    Subtracts the reference channel (Channel 0) from each EEG channel.
    
    Args:
        eeg_segment (np.ndarray): EEG data (n_channels, n_samples), excluding the reference channel.
        reference_channel (np.ndarray): The reference channel data (1D array of n_samples).
    
    Returns:
        np.ndarray: The EEG data with the reference channel subtracted.
    """
    return reference_channel - eeg_segment


#########
# MAIN
#########
def main():
    # Initialize Streaming Board
    board = BrainFlowBoardSetup(board_id = BoardIds.PLAYBACK_FILE_BOARD.value,
                                file = '120s_cyton_recording.csv', # ~120s recording where participant looked at each stimulus for 15 seconds before switching clock-wise to the next
                                master_board = BoardIds.CYTON_BOARD.value )
    board.setup()
    
    # Run the SSVEP Stimulus in a separate process
    stimulus_process = SSVEPStimulusRunner(box_frequencies=frequencies, 
                                            box_texts=buttons, 
                                            box_text_indices=button_pos,
                                            display_index=display,
                                            display_mode='both')
    # stimulus_process.start()

    # Wait for the SSVEP stimulus to stabilize
    # time.sleep(10)

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
    
    time.sleep(15)

    while True:
        segment = board.get_current_board_data(num_samples=n_samples)
        eeg_segment = segment[1:9, :]  # Assuming first 8 channels are EEG
        
        dc_offset_removed = remove_dc_offset(eeg_segment)
        
        visualize_all_channels_plotly(eeg_segment, dc_offset_removed)
        
        # Apply bandpass filter
        filtered_segment = filter_obj.bandpass_filter(eeg_segment, highcut=30, lowcut=0.1, order=4)

        # visualize_all_channels_plotly(eeg_segment, filtered_segment)
        
        # Check signal alignment for the first frequency
        reference_signal = cca_classifier.reference_signals[0]  # First frequency reference
        check_signal_alignment(filtered_segment, reference_signal)

        # Compare one EEG channel (Channel 1) against reference signal for the first frequency
        visualize_reference_vs_eeg(filtered_segment, reference_signal, actual_freqs[0])
        
        cca_classifier.plot_reference_signals(eeg_segment=filtered_segment)

        # Perform classification using the filtered EEG data
        correlations = []
        for freq_idx, freq in enumerate(actual_freqs):
            detected_freq, correlation = cca_classifier(filtered_segment)
            correlations.append(correlation)
            print(f"Frequency: {freq} Hz, Correlation: {correlation}")

        # Plot correlation coefficients across frequencies (optional, uncomment if needed)
        plot_correlation_across_frequencies(actual_freqs, correlations)

        # Wait for the next segment of data
        time.sleep(segment_duration)


if __name__ == '__main__':
    main()