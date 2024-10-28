import sys
import os
import time
import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, lfilter
from scipy import signal
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import neurokit2 as nk
import plotly.graph_objs as go
from plotly.subplots import make_subplots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import *
from modules.filtering import *
from modules.segmentation import *
# from modules.classification import *
from modules.ssvep_stim import *

# Setting variables:
board_id = BoardIds.CYTON_BOARD.value # BoardIds.SYNTHETIC_BOARD.value 
frequencies = [9.25, 11.25, 13.25, 15.25]
buttons = ['Right', 'Left', 'Up', 'Down']
button_pos = [0, 2, 3, 1]
display = 0
segment_duration = 3

# Static Variables - Probably don't need to touch :)
harmonics = 3 # np.arange(1, 4) # Generates the 1st, 2nd, & 3rd Harmonics
sampling_rate = BoardShim.get_sampling_rate(board_id)
n_samples = sampling_rate * segment_duration

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


#########
# Classification
#########
class SSVEPClassifier:
    def __init__(self, freqs, win_len, s_rate, n_harmonics=2):
        self.freqs = freqs
        self.win_len = win_len
        self.s_rate = s_rate
        self.n_harmonics = n_harmonics
        self.train_data = self._init_train_data()
        self.cca = CCA(n_components=1)
    
    def _init_train_data(self):
        t_vec = np.linspace(0, self.win_len, int(self.s_rate * self.win_len))
        targets = {}
        for freq in self.freqs:
            sig_sin, sig_cos = [], []
            for harmonics in range(1, self.n_harmonics + 1):
                sig_sin.append(np.sin(2 * np.pi * harmonics * freq * t_vec))
                sig_cos.append(np.cos(2 * np.pi * harmonics * freq * t_vec))
                
            signals = np.array(sig_sin + sig_cos).T
            
            scaled_signals = StandardScaler().fit_transform(signals)
                
            targets[freq] = scaled_signals # np.array(sig_sin + sig_cos).T
        return targets

    def apply_cca(self, eeg):
        """Apply CCA analysis to EEG data and return scores for each target frequency

        Args:
            eeg (np.array): EEG array [n_samples, n_chan]

        Returns:
            list of scores for target frequencies
        """
        
        eeg = StandardScaler().fit_transform(eeg)
        # self.train_data = StandardScaler().fit_transform(self.train_data)
        
        scores = []
        for key in self.train_data:
            sig_c, t_c = self.cca.fit_transform(eeg, self.train_data[key])
            scores.append(np.corrcoef(sig_c.T, t_c.T)[0, 1])
        return scores

#########
# MAIN
#########
def main():
    # Initialize Streaming Board
    board = BrainFlowBoardSetup(board_id = BoardIds.PLAYBACK_FILE_BOARD.value,
                                file = '120s_cyton_recording.csv', # ~120s recording where participant looked at each stimulus for 15 seconds before switching clock-wise to the next
                                master_board = BoardIds.CYTON_BOARD.value )
    board.setup()

    actual_freqs = [9.23, 11.43, 13.33, 15.0]
    # 240 Hz = [9.23, 11.43, 13.33, 15.0]
    # 239 Hz = [9.19, 11.38, 13.28, 14.94]
    print("Actual Frequencies:", actual_freqs)

    # cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
    #                                 harmonics=harmonics, 
    #                                 sampling_rate=sampling_rate, 
    #                                 n_samples=n_samples, 
    #                                 method='CCA', 
    #                                 stack_harmonics=True)
    
    cca_classifier = SSVEPClassifier(freqs=actual_freqs, 
                                     win_len=segment_duration, 
                                     s_rate=sampling_rate)

    filter_obj = Filtering(sampling_rate)
    
    time.sleep(15)

    while True:
        segment = board.get_current_board_data(num_samples=n_samples)
        eeg_segment = segment[1:9, :]  # Channels 1-9 are EEG channels
        
        dc_offset_removed = remove_dc_offset(eeg_segment)
                
        # Apply bandpass filter
        filtered_segment = filter_obj.bandpass_filter(dc_offset_removed, highcut=30, lowcut=0.1, order=4)

        r = cca_classifier.apply_cca(filtered_segment.T) 
        print(r)
        
        # visualize_all_channels_plotly(eeg_segment, filtered_segment)
        
        # Check signal alignment for the first frequency
        # reference_signal = cca_classifier.reference_signals[0]  # First frequency reference
        # check_signal_alignment(filtered_segment, reference_signal)

        # Compare one EEG channel (Channel 1) against reference signal for the first frequency
        # visualize_reference_vs_eeg(filtered_segment, reference_signal, actual_freqs[0])
        
        

        # Perform classification using the filtered EEG data
        # correlations = []
        # for freq_idx, freq in enumerate(actual_freqs):
        #     detected_freq, correlation = cca_classifier(filtered_segment)
        #     correlations.append(correlation)
        #     print(f"Frequency: {freq} Hz, Correlation: {correlation}")

        # Plot correlation coefficients across frequencies (optional, uncomment if needed)
        # plot_correlation_across_frequencies(actual_freqs, correlations)
        
    

        # Wait for the next segment of data
        time.sleep(segment_duration)


if __name__ == '__main__':
    main()