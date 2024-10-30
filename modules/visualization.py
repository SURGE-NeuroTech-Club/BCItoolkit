import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import mne

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


def compare_eeg_time_series(data_before, data_after, sampling_rate, channel_names=None):
    """
    Compare two sets of EEG data by plotting them side by side (before vs after).
    
    Parameters:
    - data_before: np.array, shape (n_channels, n_samples), the original EEG data
    - data_after: np.array, shape (n_channels, n_samples), the processed EEG data
    - sampling_rate: int, the sampling rate of the data in Hz
    - channel_names: list of strings, optional names for each channel
    """
    n_channels, n_samples = data_before.shape
    time = np.arange(0, n_samples) / sampling_rate

    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 2 * n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]  # Make it iterable if only one channel

    for i in range(n_channels):
        # Plot before data (left column)
        axes[i, 0].plot(time, data_before[i], label=f'Before: Channel {i+1}' if not channel_names else f'Before: {channel_names[i]}')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_title('Before')
        axes[i, 0].legend(loc='upper right')

        # Plot after data (right column)
        axes[i, 1].plot(time, data_after[i], label=f'After: Channel {i+1}' if not channel_names else f'After: {channel_names[i]}', color='orange')
        axes[i, 1].set_title('After')
        axes[i, 1].legend(loc='upper right')

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def compare_psd(data_before, data_after, sampling_rate, channel_names=None):
    """
    Compare the Power Spectral Density (PSD) of two EEG datasets.
    
    Parameters:
    - data_before: np.array, shape (n_channels, n_samples)
    - data_after: np.array, shape (n_channels, n_samples)
    - sampling_rate: int, the sampling rate of the data in Hz
    - channel_names: list of strings, optional names for each channel
    """
    n_channels = data_before.shape[0]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i in range(n_channels):
        freqs_before, psd_before = welch(data_before[i], fs=sampling_rate, nperseg=1024)
        freqs_after, psd_after = welch(data_after[i], fs=sampling_rate, nperseg=1024)
        
        ax.semilogy(freqs_before, psd_before, label=f'Before: {channel_names[i]}' if channel_names else f'Before: Channel {i+1}', alpha=0.6)
        ax.semilogy(freqs_after, psd_after, label=f'After: {channel_names[i]}' if channel_names else f'After: Channel {i+1}', linestyle='--', alpha=0.8)
    
    ax.set_title('Comparison of Power Spectral Density (PSD)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (uV^2/Hz)')
    ax.legend(loc='upper right')
    plt.show()


def compare_psd_side_by_side(data_before, data_after, sampling_rate, channel_names=None):
    """
    Compare the Power Spectral Density (PSD) of two EEG datasets side by side (before vs after).
    
    Parameters:
    - data_before: np.array, shape (n_channels, n_samples), the original EEG data.
    - data_after: np.array, shape (n_channels, n_samples), the processed EEG data.
    - sampling_rate: int, the sampling rate of the data in Hz.
    - channel_names: list of strings, optional names for each channel.
    """
    n_channels = data_before.shape[0]
    
    # Create figure with 2 columns (Before and After) for each channel
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3 * n_channels), sharex=True, sharey=True)

    if n_channels == 1:
        axes = [axes]  # Ensure axes are iterable for single channel
    
    for i in range(n_channels):
        # Compute PSD for "Before" data
        freqs_before, psd_before = welch(data_before[i], fs=sampling_rate, nperseg=1024)
        freqs_after, psd_after = welch(data_after[i], fs=sampling_rate, nperseg=1024)
        
        # Plot Before PSD (left column)
        axes[i, 0].semilogy(freqs_before, psd_before, label=f'Before: Channel {i+1}' if not channel_names else f'Before: {channel_names[i]}', color='b')
        axes[i, 0].set_title(f'Before: {channel_names[i]}' if channel_names else f'Before: Channel {i+1}')
        axes[i, 0].set_ylabel('Power (uV^2/Hz)')
        axes[i, 0].set_xlabel('Frequency (Hz)')
        axes[i, 0].legend(loc='upper right')

        # Plot After PSD (right column)
        axes[i, 1].semilogy(freqs_after, psd_after, label=f'After: Channel {i+1}' if not channel_names else f'After: {channel_names[i]}', color='orange')
        axes[i, 1].set_title(f'After: {channel_names[i]}' if channel_names else f'After: Channel {i+1}')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def compare_psd_stacked(data_before, data_after, sampling_rate, channel_names=None, average=False):
    """
    Compare the Power Spectral Density (PSD) of two EEG datasets, showing all channels 
    for "before" in one panel and all channels for "after" in another panel, side by side.
    
    Parameters:
    - data_before: np.array, shape (n_channels, n_samples), the original EEG data.
    - data_after: np.array, shape (n_channels, n_samples), the processed EEG data.
    - sampling_rate: int, the sampling rate of the data in Hz.
    - channel_names: list of strings, optional names for each channel.
    - average: bool, whether to plot the average PSD across all channels or individual channels.
    """
    n_channels = data_before.shape[0]

    # Create a figure with 2 columns: one for Before and one for After
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(n_channels)]
    
    # If average is True, calculate the mean PSD across channels
    if average:
        # Compute the average PSD for "Before" data
        psd_before_avg = np.zeros_like(welch(data_before[0], fs=sampling_rate, nperseg=1024)[1])
        for i in range(n_channels):
            freqs_before, psd_before = welch(data_before[i], fs=sampling_rate, nperseg=1024)
            psd_before_avg += psd_before
        psd_before_avg /= n_channels

        # Plot the average PSD for "Before" data on the left panel
        axes[0].semilogy(freqs_before, psd_before_avg, label='Average', color='blue')
        axes[0].set_title('Before (Average)')
        axes[0].set_ylabel('Power (uV^2/Hz)')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].legend(loc='upper right')

        # Compute the average PSD for "After" data
        psd_after_avg = np.zeros_like(welch(data_after[0], fs=sampling_rate, nperseg=1024)[1])
        for i in range(n_channels):
            freqs_after, psd_after = welch(data_after[i], fs=sampling_rate, nperseg=1024)
            psd_after_avg += psd_after
        psd_after_avg /= n_channels

        # Plot the average PSD for "After" data on the right panel
        axes[1].semilogy(freqs_after, psd_after_avg, label='Average', color='orange')
        axes[1].set_title('After (Average)')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].legend(loc='upper right')

    else:
        # Plot all channels in the "Before" dataset on the left panel
        for i in range(n_channels):
            freqs_before, psd_before = welch(data_before[i], fs=sampling_rate, nperseg=1024)
            axes[0].semilogy(freqs_before, psd_before, label=channel_names[i], alpha=0.6)
        
        axes[0].set_title('Before (All Channels)')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Power (uV^2/Hz)')
        axes[0].legend(loc='upper right')

        # Plot all channels in the "After" dataset on the right panel
        for i in range(n_channels):
            freqs_after, psd_after = welch(data_after[i], fs=sampling_rate, nperseg=1024)
            axes[1].semilogy(freqs_after, psd_after, label=channel_names[i], linestyle='--', alpha=0.8)
        
        axes[1].set_title('After (All Channels)')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].legend(loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def compute_snr(eeg_data):
    """
    Computes Signal-to-Noise Ratio (SNR) for each EEG channel.
    Assuming that noise can be estimated by the signal's standard deviation.
    
    Parameters:
    - eeg_data: np.array, shape (n_channels, n_samples), the EEG data.
    
    Returns:
    - snr_values: np.array, SNR values for each channel.
    """
    signal_power = np.mean(np.square(eeg_data), axis=1)
    noise_power = np.var(eeg_data, axis=1)
    snr_values = 10 * np.log10(signal_power / noise_power)
    return snr_values

def compute_variance(eeg_data):
    """
    Computes the variance for each EEG channel.
    
    Parameters:
    - eeg_data: np.array, shape (n_channels, n_samples), the EEG data.
    
    Returns:
    - var_values: np.array, variance values for each channel.
    """
    return np.var(eeg_data, axis=1)

def plot_signal_quality(eeg_data, channel_names=None):
    """
    Plots signal quality indicators (SNR and Variance) for EEG data.
    
    Parameters:
    - eeg_data: np.array, shape (n_channels, n_samples), the EEG data.
    - channel_names: list of strings, optional names for each channel.
    """
    snr_values = compute_snr(eeg_data)
    variance_values = compute_variance(eeg_data)
    n_channels = eeg_data.shape[0]

    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(n_channels)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot SNR
    ax1.bar(channel_names, snr_values, color='b')
    ax1.set_title('Signal-to-Noise Ratio (SNR)')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_xticklabels(channel_names, rotation=45, ha='right')

    # Plot Variance
    ax2.bar(channel_names, variance_values, color='g')
    ax2.set_title('Signal Variance')
    ax2.set_ylabel('Variance (uV^2)')
    ax2.set_xticklabels(channel_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
