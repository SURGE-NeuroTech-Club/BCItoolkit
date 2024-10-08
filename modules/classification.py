import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
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
        reference_signals = {}
        time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
        for freq in self.frequencies:
            Yn = np.vstack([np.sin(2 * np.pi * harmon * freq * time) for harmon in self.harmonics] + \
                           [np.cos(2 * np.pi * harmon * freq * time) for harmon in self.harmonics])
            reference_signals[freq] = Yn.T
        return reference_signals


    def _cca_analysis(self, eeg_data, reference_signal):
        """
        Performs Canonical Correlation Analysis (CCA) between EEG data and reference signals.
        Uses 1 component.
        """
        cca = CCA(n_components=1) 
        Xs_scores = cca.fit_transform(eeg_data.T, reference_signal)
        corr = np.corrcoef(Xs_scores[0][:, 0], Xs_scores[1][:, 0])[0, 1]
        return corr

    def __call__(self, eeg_segment):
        """
        Classifies the EEG data using CCA.
        
        Args:
            eeg_segment (np.ndarray): The EEG data to be analyzed (n_channels, n_samples).
        
        Returns:
            tuple: The detected frequency and the corresponding correlation value.
        """
        max_corr, target_freq = 0, None

        for freq, ref in self.reference_signals.items():
            corr = self._cca_analysis(eeg_segment, ref)
            if corr > max_corr:
                max_corr, target_freq = corr, freq

        return target_freq, max_corr
