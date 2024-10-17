# import numpy as np
# from sklearn.cross_decomposition import CCA
# from sklearn.preprocessing import StandardScaler
# from scipy.signal import butter, filtfilt
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt

# class SSVEPClassifier:
#     """
#     A class for SSVEP classification using Canonical Correlation Analysis (CCA), Filter Bank CCA (FBCCA), or Frequency-Optimized CCA (foCCA).
#     """

#     def __init__(self, frequencies, harmonics, sampling_rate, n_samples, method='CCA', num_subbands=5, stack_harmonics=True):
#         """
#         Initializes the SSVEPClassifier.

#         Args:
#             frequencies (list): List of target frequencies.
#             harmonics (list): List of harmonics to generate for each frequency.
#             sampling_rate (float): The sampling rate of the EEG data.
#             n_samples (int): The number of samples in the time window for analysis.
#             method (str): The method to use ('CCA', 'FBCCA', or 'foCCA').
#             num_subbands (int): The number of subbands for filtering the data (used only for FBCCA).
#             stack_harmonics (bool): Whether to stack harmonics for reference signals.
#         """
#         self.frequencies = frequencies
#         self.harmonics = harmonics
#         self.sampling_rate = sampling_rate
#         self.n_samples = n_samples
#         self.method = method
#         self.num_subbands = num_subbands
#         self.stack_harmonics = stack_harmonics
#         self.reference_signals = self._generate_reference_signals()

#     def _generate_reference_signals(self):
#         """
#         Generates reference signals (sine and cosine waves) for each target frequency and its harmonics.
#         """
#         reference_signals = {}
#         time = np.linspace(1/self.sampling_rate, self.n_samples/self.sampling_rate, self.n_samples)#np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
#         for freq in self.frequencies:
#             Yn = np.vstack([np.sin(2 * np.pi * harmon * freq * time) for harmon in self.harmonics] + \
#                            [np.cos(2 * np.pi * harmon * freq * time) for harmon in self.harmonics])
#             reference_signals[freq] = Yn.T
        
#         # Yn_list = []
#         # for freq_idx in range(len(self.frequencies)):
#         #     Yn = None
#         #     freq = self.frequencies[int(freq_idx)]
#         #     time = np.linspace(1/self.sampling_rate, self.n_samples/self.sampling_rate, self.n_samples)

#         #     for harmon in self.harmonics:
#         #         if Yn is None:
#         #             Yn = np.vstack((np.sin(2*np.pi*harmon*freq*time), np.cos(2*np.pi*harmon*freq*time)))
#         #         else:
#         #             Yn = np.vstack((Yn, np.sin(2*np.pi*harmon*freq*time), np.cos(2*np.pi*harmon*freq*time)))
#         #     Yn_list.append(Yn)
        
#         # reference_signals = Yn_list
        
#         return reference_signals

#     def _scale_signals(self, eeg_data, reference_signal):
#         """
#         Standardizes the EEG data and reference signals to have zero mean and unit variance.
        
#         Args:
#             eeg_data (np.ndarray): The EEG data (n_channels, n_samples).
#             reference_signal (np.ndarray): The reference signals (n_samples, n_features).
        
#         Returns:
#             np.ndarray, np.ndarray: Standardized EEG data and reference signals.
#         """
#         scaler_eeg = StandardScaler()
#         scaler_ref = StandardScaler()
        
#         # Standardize EEG data (n_channels, n_samples) -> Transpose to (n_samples, n_channels) for scaling
#         eeg_standardized = scaler_eeg.fit_transform(eeg_data.T).T
        
#         # Standardize reference signals
#         reference_standardized = scaler_ref.fit_transform(reference_signal)
        
#         return eeg_standardized, reference_standardized

#     def _cca_analysis(self, eeg_data, reference_signal):
#         """
#         Performs Canonical Correlation Analysis (CCA) between EEG data and reference signals.
#         Uses 1 component.
#         """
#         # Scale the EEG data and reference signal
#         eeg_scaled, ref_scaled = self._scale_signals(eeg_data, reference_signal)

#         # Perform CCA with 1 component
#         X_len = eeg_data.shape[1]  # Should be n_samples
#         cca = CCA(n_components=1)
#         Xs_scores = cca.fit_transform(eeg_scaled.T, ref_scaled[-X_len:, :])  # Transpose to match samples/features
#         corr = np.corrcoef(Xs_scores[0][:, 0], Xs_scores[1][:, 0])[0, 1]
#         return corr

#     def __call__(self, eeg_segment):
#         """
#         Classifies the EEG data using CCA.
        
#         Args:
#             eeg_segment (np.ndarray): The EEG data to be analyzed (n_channels, n_samples).
        
#         Returns:
#             tuple: The detected frequency and the corresponding correlation value.
#         """
#         max_corr, target_freq = 0, None

#         for freq, ref in self.reference_signals.items():
#             corr = self._cca_analysis(eeg_segment, ref)
#             if corr > max_corr:
#                 max_corr, target_freq = corr, freq

#         return target_freq, max_corr






# ### *--- Old SSVEPClassifier ---*

# # import numpy as np
# # from sklearn.cross_decomposition import CCA
# # from scipy.signal import butter, filtfilt
# # from scipy.optimize import minimize
# # import matplotlib.pyplot as plt

# # class SSVEPClassifier:
# #     """
# #     A class for SSVEP classification using Canonical Correlation Analysis (CCA), Filter Bank CCA (FBCCA), or Frequency-Optimized CCA (foCCA).
# #     """

# #     def __init__(self, frequencies, harmonics, sampling_rate, n_samples, method='CCA', num_subbands=5, stack_harmonics=True):
# #         """
# #         Initializes the SSVEPClassifier.

# #         Args:
# #             frequencies (list): List of target frequencies.
# #             harmonics (list): List of harmonics to generate for each frequency.
# #             sampling_rate (float): The sampling rate of the EEG data.
# #             n_samples (int): The number of samples in the time window for analysis.
# #             method (str): The method to use ('CCA', 'FBCCA', or 'foCCA').
# #             num_subbands (int): The number of subbands for filtering the data (used only for FBCCA).
# #             stack_harmonics (bool): Whether to stack harmonics for reference signals.
# #         """
# #         self.frequencies = frequencies
# #         self.harmonics = harmonics
# #         self.sampling_rate = sampling_rate
# #         self.n_samples = n_samples
# #         self.method = method
# #         self.num_subbands = num_subbands
# #         self.stack_harmonics = stack_harmonics
# #         self.reference_signals = self._generate_reference_signals()

# #     def _generate_reference_signals(self):
# #         """
# #         Generates reference signals (sine and cosine waves) for each target frequency and its harmonics.
# #         """
# #         reference_signals = {}
# #         time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
# #         for freq in self.frequencies:
# #             Yn = np.vstack([np.sin(2 * np.pi * harmon * freq * time) for harmon in self.harmonics] + \
# #                            [np.cos(2 * np.pi * harmon * freq * time) for harmon in self.harmonics])
# #             reference_signals[freq] = Yn.T
# #         return reference_signals


# #     def _cca_analysis(self, eeg_data, reference_signal):
# #         """
# #         Performs Canonical Correlation Analysis (CCA) between EEG data and reference signals.
# #         Uses 1 component.
# #         """
# #         X_len = eeg_data.shape[1] # should be n_samples
# #         cca = CCA(n_components=1) 
# #         Xs_scores = cca.fit_transform(eeg_data.T, reference_signal[-X_len:, :]) # , reference_signal)
# #         corr = np.corrcoef(Xs_scores[0][:, 0], Xs_scores[1][:, 0])[0, 1]
# #         return corr

# #     def __call__(self, eeg_segment):
# #         """
# #         Classifies the EEG data using CCA.
        
# #         Args:
# #             eeg_segment (np.ndarray): The EEG data to be analyzed (n_channels, n_samples).
        
# #         Returns:
# #             tuple: The detected frequency and the corresponding correlation value.
# #         """
# #         max_corr, target_freq = 0, None

# #         for freq, ref in self.reference_signals.items():
# #             corr = self._cca_analysis(eeg_segment, ref)
# #             if corr > max_corr:
# #                 max_corr, target_freq = corr, freq

# #         return target_freq, max_corr
    
    
    
    
# # ## *--- Backup before I change stuff with harmonics --*    
    
    
# # # import numpy as np
# # # from sklearn.cross_decomposition import CCA
# # # from scipy.signal import butter, filtfilt
# # # from scipy.optimize import minimize
# # # import matplotlib.pyplot as plt

# # # class SSVEPClassifier:
# # #     """
# # #     A class for SSVEP classification using Canonical Correlation Analysis (CCA), Filter Bank CCA (FBCCA), or Frequency-Optimized CCA (foCCA).
# # #     """

# # #     def __init__(self, frequencies, harmonics, sampling_rate, n_samples, method='CCA', num_subbands=5, stack_harmonics=True):
# # #         """
# # #         Initializes the SSVEPClassifier.

# # #         Args:
# # #             frequencies (list): List of target frequencies.
# # #             harmonics (list): List of harmonics to generate for each frequency.
# # #             sampling_rate (float): The sampling rate of the EEG data.
# # #             n_samples (int): The number of samples in the time window for analysis.
# # #             method (str): The method to use ('CCA', 'FBCCA', or 'foCCA').
# # #             num_subbands (int): The number of subbands for filtering the data (used only for FBCCA).
# # #             stack_harmonics (bool): Whether to stack harmonics for reference signals.
# # #         """
# # #         self.frequencies = frequencies
# # #         self.harmonics = harmonics
# # #         self.sampling_rate = sampling_rate
# # #         self.n_samples = n_samples
# # #         self.method = method
# # #         self.num_subbands = num_subbands
# # #         self.stack_harmonics = stack_harmonics
# # #         self.reference_signals = self._generate_reference_signals()

# # #     def _generate_reference_signals(self):
# # #         """
# # #         Generates reference signals (sine and cosine waves) for each target frequency and its harmonics.
# # #         """
# # #         reference_signals = {}
# # #         time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
# # #         for freq in self.frequencies:
# # #             Yn = np.vstack([np.sin(2 * np.pi * harmon * freq * time) for harmon in self.harmonics] + \
# # #                            [np.cos(2 * np.pi * harmon * freq * time) for harmon in self.harmonics])
# # #             reference_signals[freq] = Yn.T
# # #         return reference_signals


# # #     def _cca_analysis(self, eeg_data, reference_signal):
# # #         """
# # #         Performs Canonical Correlation Analysis (CCA) between EEG data and reference signals.
# # #         Uses 1 component.
# # #         """
# # #         X_len = eeg_data.shape[1] # should be n_samples
# # #         cca = CCA(n_components=1) 
# # #         Xs_scores = cca.fit_transform(eeg_data.T, reference_signal[-eeg_data:, :]) # , reference_signal)
# # #         corr = np.corrcoef(Xs_scores[0][:, 0], Xs_scores[1][:, 0])[0, 1]
# # #         return corr

# # #     def __call__(self, eeg_segment):
# # #         """
# # #         Classifies the EEG data using CCA.
        
# # #         Args:
# # #             eeg_segment (np.ndarray): The EEG data to be analyzed (n_channels, n_samples).
        
# # #         Returns:
# # #             tuple: The detected frequency and the corresponding correlation value.
# # #         """
# # #         max_corr, target_freq = 0, None

# # #         for freq, ref in self.reference_signals.items():
# # #             corr = self._cca_analysis(eeg_segment, ref)
# # #             if corr > max_corr:
# # #                 max_corr, target_freq = corr, freq

# # #         return target_freq, max_corr

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class SSVEPClassifier:
    """
    A class for SSVEP classification using Canonical Correlation Analysis (CCA), Filter Bank CCA (FBCCA), or Frequency-Optimized CCA (foCCA).
    """

    def __init__(self, frequencies, harmonics, sampling_rate, n_samples, method='CCA', num_subbands=5, stack_harmonics=True):
        """
        Initializes the SSVEPClassifier.

        Args:
            frequencies (list): List of target frequencies.
            harmonics (list): List of harmonics to generate for each frequency.
            sampling_rate (float): The sampling rate of the EEG data.
            n_samples (int): The number of samples in the time window for analysis.
            method (str): The method to use ('CCA', 'FBCCA', or 'foCCA').
            num_subbands (int): The number of subbands for filtering the data (used only for FBCCA).
            stack_harmonics (bool): Whether to stack harmonics for reference signals.
        """
        self.frequencies = frequencies
        self.harmonics = harmonics
        self.sampling_rate = sampling_rate
        self.n_samples = n_samples
        self.method = method
        self.num_subbands = num_subbands
        self.stack_harmonics = stack_harmonics
        self.reference_signals = self._generate_reference_signals()

    def _generate_reference_signals(self):
        """
        Generates reference signals (sine and cosine waves) for each target frequency and its harmonics.
        """
        Yn_list = []
        time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)

        for freq in self.frequencies:
            Yn = np.vstack([np.sin(2 * np.pi * harmon * freq * time) for harmon in range(self.harmonics)] + 
                           [np.cos(2 * np.pi * harmon * freq * time) for harmon in range(self.harmonics)])
            Yn_list.append(Yn.T)  # Stacking both sine and cosine components
        return Yn_list

    def _scale_signals(self, eeg_data, reference_signal):
        """
        Standardizes the EEG data and reference signals to have zero mean and unit variance.
        """
        scaler_eeg = StandardScaler()
        scaler_ref = StandardScaler()

        # Standardize EEG data (n_channels, n_samples) -> Transpose to (n_samples, n_channels) for scaling
        eeg_standardized = scaler_eeg.fit_transform(eeg_data.T).T

        # Standardize reference signals
        reference_standardized = scaler_ref.fit_transform(reference_signal)

        return eeg_standardized, reference_standardized

    def _cca_analysis(self, eeg_data, reference_signal):
        """
        Performs Canonical Correlation Analysis (CCA) between EEG data and reference signals.
        """
        eeg_scaled, ref_scaled = self._scale_signals(eeg_data, reference_signal)
        X_len = eeg_data.shape[1]  # Number of samples

        cca = CCA(n_components=1)
        Xs_scores = cca.fit_transform(eeg_scaled.T, ref_scaled[:X_len, :])  # Transpose to match samples/features
        corr = np.corrcoef(Xs_scores[0][:, 0], Xs_scores[1][:, 0])[0, 1]
        return corr

    def __call__(self, eeg_segment):
        """
        Classifies the EEG data using CCA.
        """
        max_corr, target_freq = 0, None

        for freq_idx, ref in enumerate(self.reference_signals):
            corr = self._cca_analysis(eeg_segment, ref)
            if corr > max_corr:
                max_corr, target_freq = corr, self.frequencies[freq_idx]

        return target_freq, max_corr

    def plot_reference_signals(self, eeg_segment, channel_idx=0, freq_idx=0):
        """
        Plots the EEG signal and the corresponding reference signals (sine and cosine) for debugging.

        Args:
            eeg_segment (np.ndarray): The EEG data segment (n_channels, n_samples).
            channel_idx (int): The channel index of the EEG data to plot.
            freq_idx (int): The index of the frequency to generate and plot reference signals for.
        """
        time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
        eeg_signal = eeg_segment[channel_idx, :]
        reference_signals = self.reference_signals[freq_idx]
        freq = self.frequencies[freq_idx]

        plt.figure(figsize=(12, 6))

        # Plot EEG signal
        plt.plot(time, eeg_signal, label=f'EEG Channel {channel_idx+1}', linewidth=2)

        # Plot reference signals (first sine and cosine harmonic for simplicity)
        plt.plot(time, reference_signals[:, 0], '--', label=f'Reference Sin (f={freq} Hz)', alpha=0.8)
        plt.plot(time, reference_signals[:, len(self.harmonics)], '--', label=f'Reference Cos (f={freq} Hz)', alpha=0.8)

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'EEG Signal vs Reference Signals for f = {freq} Hz')
        plt.legend()
        plt.grid(True)
        plt.show()

# # Example Usage
# frequencies = [10, 12, 15]  # Example frequencies in Hz
# harmonics = [1, 2, 3]  # Harmonics to include
# sampling_rate = 250  # Example sampling rate in Hz
# n_samples = 500  # Example number of samples

# # Simulate a random EEG signal (8 channels, 500 samples)
# np.random.seed(42)
# eeg_segment = np.random.randn(8, n_samples)

# # Initialize and test the SSVEPClassifier
# classifier = SSVEPClassifier(frequencies, harmonics, sampling_rate, n_samples)

# # Plot reference signals and compare with EEG signal for debugging
# classifier.plot_reference_signals(eeg_segment, channel_idx=0, freq_idx=0)

# # Perform classification and print the result
# detected_freq, max_corr = classifier(eeg_segment)
# print(f"Detected Frequency: {detected_freq} Hz, Maximum Correlation: {max_corr}")
