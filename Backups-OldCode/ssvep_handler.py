import numpy as np
from mvlearn.embed import CCA
from scipy.signal import welch, butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use a non-GUI backend

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


    
# import numpy as np
# from mvlearn.embed import CCA
# from scipy.signal import welch
# import matplotlib.pyplot as plt
# import matplotlib

# matplotlib.use('Agg')  # Use a non-GUI backend

# class SSVEP_SNR:
#     """
#     A class to calculate and plot Signal-to-Noise Ratio (SNR) for SSVEP signals.

#     Attributes:
#         signal (np.ndarray): The input EEG signal.
#         fs (float): The sampling frequency of the signal.
#         noise_bandwidth (float): The bandwidth around each frequency considered as noise.
#     """

#     def __init__(self, signal, fs, noise_bandwidth=1):
#         """
#         Initializes the SSVEP_SNR class with the given parameters.

#         Args:
#             signal (np.ndarray): The input EEG signal.
#             fs (float): The sampling frequency of the signal.
#             noise_bandwidth (float): The bandwidth around each frequency considered as noise.
#         """
#         self.signal = signal
#         self.fs = fs
#         self.noise_bandwidth = noise_bandwidth

#     def calculate_psd(self):
#         """
#         Calculates the Power Spectral Density (PSD) of the signal.

#         Returns:
#             tuple: Arrays of frequencies and corresponding PSD values.
#         """
#         freqs, psd = welch(self.signal, self.fs, nperseg=1024)
#         return freqs, psd

#     def calculate_snr(self):
#         """
#         Calculates the Signal-to-Noise Ratio (SNR) of the signal for all frequencies.

#         Returns:
#             tuple: Arrays of frequencies and corresponding SNR values.
#         """
#         freqs, psd = self.calculate_psd()
#         snr = np.zeros_like(psd)

#         for i in range(len(freqs)):
#             noise_range = np.logical_and(freqs >= freqs[i] - self.noise_bandwidth, freqs <= freqs[i] + self.noise_bandwidth)
#             noise_range[i] = False  # Exclude the signal frequency itself
#             signal_power = psd[i]
#             noise_power = np.mean(psd[noise_range])
#             snr[i] = 10 * np.log10(signal_power / noise_power)
        
#         return freqs, snr

#     def plot_snr(self, filename='snr_plot.png', fmin=1.0, fmax=50.0):
#         """
#         Plots the PSD and SNR of the signal and saves the plot to a file.

#         Args:
#             filename (str): The name of the file to save the plot.
#             fmin (float): The minimum frequency for the plot.
#             fmax (float): The maximum frequency for the plot.
#         """
#         freqs, psd = self.calculate_psd()
#         freqs, snr = self.calculate_snr()

#         freq_range = range(
#             np.where(np.floor(freqs) == fmin)[0][0], np.where(np.ceil(freqs) == fmax)[0][0]
#         )
        
#         psd_db = 10 * np.log10(psd)
        
#         fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))

#         # Plot PSD
#         axes[0].plot(freqs[freq_range], psd_db[freq_range], color="b")
#         axes[0].fill_between(
#             freqs[freq_range], psd_db[freq_range], color="b", alpha=0.2
#         )
#         axes[0].set(title="PSD Spectrum", ylabel="Power Spectral Density [dB]")

#         # Plot SNR
#         axes[1].plot(freqs[freq_range], snr[freq_range], color="r")
#         axes[1].fill_between(
#             freqs[freq_range], snr[freq_range], color="r", alpha=0.2
#         )
#         axes[1].set(
#             title="SNR Spectrum",
#             xlabel="Frequency [Hz]",
#             ylabel="SNR [dB]",
#             ylim=[-2, 30],
#             xlim=[fmin, fmax],
#         )

#         plt.tight_layout()
#         plt.savefig(filename)
#         plt.close()


# class ClassifySSVEP:
#     """
#     A class to generate reference signals and perform Canonical Correlation Analysis (CCA) for SSVEP classification.

#     Attributes:
#         frequencies (list): List of target frequencies.
#         harmonics (list): List of harmonics to generate for each frequency.
#         sampling_rate (float): The sampling rate of the EEG data.
#         n_samples (int): The number of samples in the time window for analysis.
#         stack_harmonics (bool): Whether to stack harmonics for reference signals.
#         reference_signals (dict): Dictionary of reference signals for each frequency.
#     """

#     def __init__(self, frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=True):
#         """
#         Initializes the ClassifySSVEP class with the given parameters.

#         Args:
#             frequencies (list): List of target frequencies.
#             harmonics (list): List of harmonics to generate for each frequency.
#             sampling_rate (float): The sampling rate of the EEG data.
#             n_samples (int): The number of samples in the time window for analysis.
#             stack_harmonics (bool): Whether to stack harmonics for reference signals.
#         """
#         self.frequencies = frequencies
#         self.harmonics = harmonics
#         self.sampling_rate = sampling_rate
#         self.n_samples = n_samples
#         self.stack_harmonics = stack_harmonics
#         self.reference_signals = self._generate_reference_signals()

#     def _generate_reference_signals(self):
#         """
#         Generates reference signals (sine and cosine waves) for each target frequency and its harmonics.

#         Returns:
#             dict: A dictionary containing the generated reference signals for each frequency.
#         """
#         reference_signals = {}
#         time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
#         for freq in self.frequencies:
#             signals = []
#             for harmon in self.harmonics:
#                 sine_wave = np.sin(2 * np.pi * harmon * freq * time)
#                 cosine_wave = np.cos(2 * np.pi * harmon * freq * time)
#                 signals.append(sine_wave)
#                 signals.append(cosine_wave)
#             if self.stack_harmonics:
#                 reference_signals[freq] = np.vstack(signals).T
#             else:
#                 reference_signals[freq] = np.array(signals)
#         return reference_signals

#     def get_reference_signals(self, frequency):
#         """
#         Retrieves the reference signals for a given frequency.

#         Args:
#             frequency (float): The target frequency.

#         Returns:
#             np.ndarray: The reference signals for the given frequency.
#         """
#         return self.reference_signals.get(frequency, None)

#     def cca_analysis(self, eeg_data):
#         """
#         Performs Canonical Correlation Analysis (CCA) to identify the target frequency the user is focusing on.

#         Args:
#             eeg_data (np.ndarray): The EEG data to be analyzed.
        
#         Returns:
#             tuple: The detected frequency and the corresponding correlation value.
#         """
#         cca = CCA(n_components=1)
#         max_corr = 0
#         target_freq = None

#         for freq, ref in self.reference_signals.items():
#             if self.stack_harmonics:
#                 if eeg_data.shape[1] != ref.shape[0]:
#                     raise ValueError("EEG data and reference signals must have the same number of samples")
#                 cca.fit([eeg_data.T, ref])
#                 U, V = cca.transform([eeg_data.T, ref])
#             else:
#                 U, V = None, None
#                 if eeg_data.shape[1] != ref.shape[1]:
#                     raise ValueError("EEG data and reference signals must have the same number of samples")
#                 for i in range(ref.shape[0] // 2):
#                     cca.fit([eeg_data.T, ref[2*i:2*i+2, :].T])
#                     U_tmp, V_tmp = cca.transform([eeg_data.T, ref[2*i:2*i+2, :].T])
#                     if U is None:
#                         U, V = U_tmp, V_tmp
#                     else:
#                         U = np.hstack((U, U_tmp))
#                         V = np.hstack((V, V_tmp))
#             corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
#             if corr > max_corr:
#                 max_corr = corr
#                 target_freq = freq
#         return target_freq, max_corr

#     def check_snr(self, eeg_data):
#         """
#         Calculates the SNR for each target frequency.

#         Args:
#             eeg_data (np.ndarray): The EEG data to be analyzed.

#         Returns:
#             dict: SNR values for each target frequency.
#         """
#         signal = eeg_data.flatten()  # Assuming eeg_data is 2D: (n_channels, n_samples)
#         snr_calculator = SSVEP_SNR(signal, self.sampling_rate)
#         freqs, snr_values = snr_calculator.calculate_snr()
        
#         snr_results = {}
#         for freq in self.frequencies:
#             target_idx = np.argmin(np.abs(freqs - freq))
#             snr_results[freq] = snr_values[target_idx]
        
#         return snr_results

# ## Example usage
# if __name__ == "__main__":
#     frequencies = [9.25, 11.25, 13.25]
#     harmonics = np.arange(1, 4)
#     sampling_rate = 256
#     n_samples = 1025

#     # Initialize with stacking harmonics
#     ssvep_harmonics_stacked = ClassifySSVEP(frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=True)

#     # Initialize without stacking harmonics
#     ssvep_harmonics_not_stacked = ClassifySSVEP(frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=False)

#     # Example EEG data (randomly generated for illustration purposes, shape: n_channels, n_samples)
#     eeg_data = np.random.randn(6, n_samples)  # Assuming 6 channels and 1025 samples

#     # Perform CCA analysis with stacked harmonics
#     detected_freq_stacked, correlation_stacked = ssvep_harmonics_stacked.cca_analysis(eeg_data)
#     print(f"Detected frequency with stacked harmonics: {detected_freq_stacked} Hz with correlation: {correlation_stacked}")

#     # Perform CCA analysis without stacking harmonics
#     detected_freq_not_stacked, correlation_not_stacked = ssvep_harmonics_not_stacked.cca_analysis(eeg_data)
#     print(f"Detected frequency without stacked harmonics: {detected_freq_not_stacked} Hz with correlation: {correlation_not_stacked}")
    
#     # Check SNR and plot
#     snr_calculator = SSVEP_SNR(eeg_data.flatten(), sampling_rate)
#     snr_calculator.plot_snr('snr_plot.png', fmin=1.0, fmax=50.0)  # Save the plot as snr_plot.png
    
#     # Check SNR for each target frequency
#     snr_results = ssvep_harmonics_stacked.check_snr(eeg_data)
#     for freq, snr in snr_results.items():
#         print(f"Frequency: {freq} Hz, SNR: {snr:.2f} dB")
