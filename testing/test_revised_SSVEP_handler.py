import time
import numpy as np
from brainflow.board_shim import BoardShim
from scipy.signal import butter, lfilter, welch
import matplotlib.pyplot as plt
import matplotlib
from mvlearn.embed import CCA

matplotlib.use('Agg')  # Use a non-GUI backend

class PreProcess:
    """
    A class to handle preprocessing of EEG data for SSVEP BCI systems.

    Attributes:
        board (BoardShim): The BrainFlow board object for EEG data acquisition.
        segment_duration (float): The duration of each data segment in seconds.
        sampling_rate (int): The sampling rate of the EEG data.
        n_samples (int): The number of samples in each data segment.
    """

    def __init__(self, board, segment_duration):
        """
        Initializes the PreProcess class with the given parameters.

        Args:
            board (BoardShim): The BrainFlow board object for EEG data acquisition.
            segment_duration (float): The duration of each data segment in seconds.
        """
        self.board = board
        self.segment_duration = segment_duration
        self.sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        self.n_samples = int(self.sampling_rate * self.segment_duration)
  
    def get_segment(self):
        """
        Retrieves the latest segment of EEG data.

        Returns:
            np.ndarray: The latest segment of EEG data, or None if insufficient data.
        """
        data = self.board.get_current_board_data(self.n_samples)
        if data.shape[1] >= self.n_samples:
            segment = data[:, -self.n_samples:]
            return segment
        return None

    def filter_data(self, data):
        """
        Applies a bandpass filter to the EEG data.

        Args:
            data (np.ndarray): The EEG data to be filtered.

        Returns:
            np.ndarray: The filtered EEG data.
        """
        lowcut = 0.5  # Example low cut frequency in Hz
        highcut = 30.0  # Example high cut frequency in Hz
        filtered_data = np.apply_along_axis(self.bandpass_filter, 1, data, lowcut, highcut, self.sampling_rate)
        return filtered_data

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Applies a bandpass filter to a single channel of EEG data.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            lowcut (float): The low cut frequency of the filter in Hz.
            highcut (float): The high cut frequency of the filter in Hz.
            fs (int): The sampling rate of the data in Hz.
            order (int): The order of the filter.

        Returns:
            np.ndarray: The bandpass filtered EEG data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def extract_features(self, data):
        """
        Extracts features from the EEG data.

        Args:
            data (np.ndarray): The EEG data from which to extract features.

        Returns:
            np.ndarray: The extracted features, including mean and standard deviation for each channel.
        """
        features = np.vstack((data.mean(axis=1), data.std(axis=1))).T
        return features

    def save_data(self, data, filename):
        """
        Saves the EEG data to a CSV file.

        Args:
            data (np.ndarray): The EEG data to be saved.
            filename (str): The name of the file to save the data to.
        """
        np.savetxt(filename, data, delimiter=',')

class SSVEP_SNR:
    """
    A class to calculate and plot Signal-to-Noise Ratio (SNR) for SSVEP signals.
    """
    def __init__(self, signal, fs, noise_bandwidth=1):
        self.signal = signal
        self.fs = fs
        self.noise_bandwidth = noise_bandwidth

    def calculate_psd(self):
        freqs, psd = welch(self.signal, self.fs, nperseg=1024)
        return freqs, psd

    def calculate_snr(self):
        freqs, psd = self.calculate_psd()
        snr = np.zeros_like(psd)
        for i in range(len(freqs)):
            noise_range = np.logical_and(freqs >= freqs[i] - self.noise_bandwidth, freqs <= freqs[i] + self.noise_bandwidth)
            noise_range[i] = False  # Exclude the signal frequency itself
            signal_power = psd[i]
            noise_power = np.mean(psd[noise_range])
            snr[i] = 10 * np.log10(signal_power / noise_power)
        return freqs, snr

    def plot_snr(self, filename='snr_plot.png', fmin=1.0, fmax=50.0):
        freqs, psd = self.calculate_psd()
        freqs, snr = self.calculate_snr()
        freq_range = range(np.where(np.floor(freqs) == fmin)[0][0], np.where(np.ceil(freqs) == fmax)[0][0])
        psd_db = 10 * np.log10(psd)
        fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
        axes[0].plot(freqs[freq_range], psd_db[freq_range], color="b")
        axes[0].fill_between(freqs[freq_range], psd_db[freq_range], color="b", alpha=0.2)
        axes[0].set(title="PSD Spectrum", ylabel="Power Spectral Density [dB]")
        axes[1].plot(freqs[freq_range], snr[freq_range], color="r")
        axes[1].fill_between(freqs[freq_range], snr[freq_range], color="r", alpha=0.2)
        axes[1].set(title="SNR Spectrum", xlabel="Frequency [Hz]", ylabel="SNR [dB]", ylim=[-2, 30], xlim=[fmin, fmax])
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

class ClassifySSVEP:
    def __init__(self, frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=True):
        self.frequencies = frequencies
        self.harmonics = harmonics
        self.sampling_rate = sampling_rate
        self.n_samples = n_samples
        self.stack_harmonics = stack_harmonics
        self.reference_signals = self._generate_reference_signals()

    def _generate_reference_signals(self):
        reference_signals = {}
        time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
        for freq in self.frequencies:
            signals = []
            for harmon in self.harmonics:
                sine_wave = np.sin(2 * np.pi * harmon * freq * time)
                cosine_wave = np.cos(2 * np.pi * harmon * freq * time)
                signals.append(sine_wave)
                signals.append(cosine_wave)
            if self.stack_harmonics:
                reference_signals[freq] = np.vstack(signals).T
            else:
                reference_signals[freq] = np.array(signals)
        return reference_signals

    def get_reference_signals(self, frequency):
        return self.reference_signals.get(frequency, None)

    def cca_analysis(self, eeg_data):
        cca = CCA(n_components=1)
        max_corr = 0
        target_freq = None
        for freq, ref in self.reference_signals.items():
            if self.stack_harmonics:
                if eeg_data.shape[1] != ref.shape[0]:
                    raise ValueError("EEG data and reference signals must have the same number of samples")
                cca.fit([eeg_data.T, ref])
                U, V = cca.transform([eeg_data.T, ref])
            else:
                U, V = None, None
                if eeg_data.shape[1] != ref.shape[1]:
                    raise ValueError("EEG data and reference signals must have the same number of samples")
                for i in range(ref.shape[0] // 2):
                    cca.fit([eeg_data.T, ref[2*i:2*i+2, :].T])
                    U_tmp, V_tmp = cca.transform([eeg_data.T, ref[2*i:2*i+2, :].T])
                    if U is None:
                        U, V = U_tmp, V_tmp
                    else:
                        U = np.hstack((U, U_tmp))
                        V = np.hstack((V, V_tmp))
            corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
            if corr > max_corr:
                max_corr = corr
                target_freq = freq
        return target_freq, max_corr

    def check_snr(self, eeg_data):
        signal = eeg_data.flatten()  # Assuming eeg_data is 2D: (n_channels, n_samples)
        snr_calculator = SSVEP_SNR(signal, self.sampling_rate)
        freqs, snr_values = snr_calculator.calculate_snr()
        snr_results = {}
        for freq in self.frequencies:
            target_idx = np.argmin(np.abs(freqs - freq))
            snr_results[freq] = snr_values[target_idx]
        return snr_results

    def visualize_ssvep(self, eeg_data, filename='ssvep_visualization.png'):
        snr_calculator = SSVEP_SNR(eeg_data.flatten(), self.sampling_rate)
        freqs, snr = snr_calculator.calculate_snr()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, snr, label='SNR')
        for freq in self.frequencies:
            ax.axvline(freq, color='r', linestyle='--', label=f'Target Frequency: {freq} Hz')
        
        ax.set(title='SSVEP Signal Visualization', xlabel='Frequency (Hz)', ylabel='SNR (dB)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

class FBCCA:
    def __init__(self, frequencies, harmonics, sampling_rate, n_samples, num_subbands=5):
        self.frequencies = frequencies
        self.harmonics = harmonics
        self.sampling_rate = sampling_rate
        self.n_samples = n_samples
        self.num_subbands = num_subbands
        self.reference_signals = self._generate_reference_signals()
        self.filters = self._generate_filters()

    def _generate_reference_signals(self):
        reference_signals = {}
        time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
        for freq in self.frequencies:
            signals = []
            for harmon in self.harmonics:
                sine_wave = np.sin(2 * np.pi * harmon * freq * time)
                cosine_wave = np.cos(2 * np.pi * harmon * freq * time)
                signals.append(sine_wave)
                signals.append(cosine_wave)
            reference_signals[freq] = np.vstack(signals).T
        return reference_signals

    def _generate_filters(self):
        filters = []
        nyquist = 0.5 * self.sampling_rate
        low = 6 / nyquist
        high = 40 / nyquist
        subband_width = (high - low) / self.num_subbands
        for i in range(self.num_subbands):
            band = [low + i * subband_width, low + (i + 1) * subband_width]
            if band[1] > 1.0:
                band[1] = 1.0
            b, a = butter(4, band, btype='band')
            filters.append((b, a))
        return filters

    def filter_data(self, data):
        filtered_data = []
        for b, a in self.filters:
            filtered_data.append(filtfilt(b, a, data, axis=-1))
        return np.array(filtered_data)

    def fbcca_analysis(self, eeg_data):
        max_corr = 0
        target_freq = None
        for freq, ref in self.reference_signals.items():
            corr = 0
            filtered_data = self.filter_data(eeg_data)
            for subband_data in filtered_data:
                cca = CCA(n_components=1)
                cca.fit([subband_data.T, ref])
                U, V = cca.transform([subband_data.T, ref])
                subband_corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
                corr += subband_corr
            corr /= self.num_subbands
            if corr > max_corr:
                max_corr = corr
                target_freq = freq
        return target_freq, max_corr

    def visualize_ssvep(self, eeg_data, filename='fbcca_ssvep_visualization.png'):
        snr_calculator = SSVEP_SNR(eeg_data.flatten(), self.sampling_rate)
        freqs, snr = snr_calculator.calculate_snr()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, snr, label='SNR')
        for freq in self.frequencies:
            ax.axvline(freq, color='r', linestyle='--', label=f'Target Frequency: {freq} Hz')
        
        ax.set(title='FBCCA SSVEP Signal Visualization', xlabel='Frequency (Hz)', ylabel='SNR (dB)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':
    import sys
    import numpy as np
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    # from ssvep_module import PreProcess, ClassifySSVEP, FBCCA

    # Simulated EEG data generation
    def generate_synthetic_eeg(frequencies, sampling_rate, n_samples, n_channels=8):
        """
        Generates synthetic EEG data with given frequencies embedded in it.

        Args:
            frequencies (list): List of frequencies to embed in the EEG data.
            sampling_rate (int): The sampling rate of the EEG data.
            n_samples (int): The number of samples to generate.
            n_channels (int): The number of EEG channels to generate.

        Returns:
            np.ndarray: Synthetic EEG data with shape (n_channels, n_samples).
        """
        t = np.linspace(0, n_samples / sampling_rate, n_samples, endpoint=False)
        eeg_data = np.zeros((n_channels, n_samples))
        for freq in frequencies:
            signal = np.sin(2 * np.pi * freq * t) + np.random.randn(n_samples) * 0.5  # Sine wave with noise
            eeg_data += signal
        return eeg_data

    # Parameters
    sampling_rate = 256  # in Hz
    segment_duration = 4.0  # in seconds
    n_samples = int(segment_duration * sampling_rate)
    frequencies = [10, 15, 20]  # Target SSVEP frequencies

    # Generate synthetic EEG data
    eeg_data = generate_synthetic_eeg(frequencies, sampling_rate, n_samples)

    # Initialize PreProcess class
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
    preprocessor = PreProcess(board, segment_duration)

    # Pre-process the EEG data
    filtered_eeg_data = preprocessor.filter_data(eeg_data)

    # Initialize ClassifySSVEP class
    harmonics = [1, 2]
    ssvep_classifier = ClassifySSVEP(frequencies, harmonics, sampling_rate, n_samples)

    # Perform CCA analysis and visualize results
    target_freq, max_corr = ssvep_classifier.cca_analysis(filtered_eeg_data)
    print(f"ClassifySSVEP - Target Frequency: {target_freq} Hz, Max Correlation: {max_corr}")

    ssvep_classifier.visualize_ssvep(filtered_eeg_data, filename='ssvep_visualization.png')

    # Initialize FBCCA class
    fbcca_classifier = FBCCA(frequencies, harmonics, sampling_rate, n_samples)

    # Perform FBCCA analysis and visualize results
    target_freq_fbcca, max_corr_fbcca = fbcca_classifier.fbcca_analysis(filtered_eeg_data)
    print(f"FBCCA - Target Frequency: {target_freq_fbcca} Hz, Max Correlation: {max_corr_fbcca}")

    fbcca_classifier.visualize_ssvep(filtered_eeg_data, filename='fbcca_ssvep_visualization.png')