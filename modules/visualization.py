import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

def plot_eeg_time_series(eeg_data, sampling_rate, channel_names=None):
    """
    Plots time series of EEG data for multiple channels.
    
    Parameters:
    - eeg_data: np.array, shape (n_channels, n_samples)
    - sampling_rate: int, the sampling rate of the data in Hz
    - channel_names: list of strings, optional names for each channel
    """
    n_channels, n_samples = eeg_data.shape
    time = np.arange(0, n_samples) / sampling_rate
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]  # Make it iterable if only one channel

    for i, ax in enumerate(axes):
        ax.plot(time, eeg_data[i], label=f'Channel {i+1}' if not channel_names else channel_names[i])
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def plot_psd(eeg_data, sampling_rate, channel_names=None):
    """
    Plots the Power Spectral Density (PSD) for each channel.
    
    Parameters:
    - eeg_data: np.array, shape (n_channels, n_samples)
    - sampling_rate: int, the sampling rate of the data in Hz
    - channel_names: list of strings, optional names for each channel
    """
    n_channels = eeg_data.shape[0]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i in range(n_channels):
        freqs, psd = welch(eeg_data[i], fs=sampling_rate, nperseg=1024)
        ax.semilogy(freqs, psd, label=f'Channel {i+1}' if not channel_names else channel_names[i])
    
    ax.set_title('Power Spectral Density (PSD) Plot')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (uV^2/Hz)')
    ax.legend(loc='upper right')
    plt.show()



import mne
import numpy as np

def plot_topomap(eeg_data, sampling_rate, channel_names, times=None):
    """
    Plots a topographic map for the EEG data at specific time points.
    
    Parameters:
    - eeg_data: np.array, shape (n_channels, n_samples)
    - channel_names: list of strings, the names of EEG channels
    - sampling_rate: int, the sampling rate of the data in Hz
    - times: list of floats, time points (in seconds) to plot topomaps
    """
    if times is None:
        times = [0.1, 0.3, 0.5]  # Example time points

    # Create MNE info structure
    info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types='eeg')
    
    # Set montage separately
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Create an MNE Evoked object
    evoked = mne.EvokedArray(eeg_data, info)

    # Plot topomap at specified time points
    fig = evoked.plot_topomap(times=times, ch_type='eeg', show_names=True, show=True)