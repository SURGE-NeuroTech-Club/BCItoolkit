import numpy as np
from mvlearn.embed import CCA
from scipy.signal import butter, filtfilt

class SSVEPClassifier:
    """
    A class for SSVEP classification using either Canonical Correlation Analysis (CCA) or Filter Bank Canonical Correlation Analysis (FBCCA).
    """

    def __init__(self, frequencies, harmonics, sampling_rate, n_samples, method='CCA', num_subbands=5, stack_harmonics=True):
        """
        Initializes the SSVEPClassifier.

        Args:
            frequencies (list): List of target frequencies.
            harmonics (list): List of harmonics to generate for each frequency.
            sampling_rate (float): The sampling rate of the EEG data.
            n_samples (int): The number of samples in the time window for analysis.
            method (str): The method to use ('CCA' or 'FBCCA').
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

        if method == 'FBCCA':
            self.filters = self._generate_filters()

    def _generate_reference_signals(self):
        """
        Generates reference signals (sine and cosine waves) for each target frequency and its harmonics.

        Returns:
            dict: A dictionary containing the generated reference signals for each frequency.
        """
        reference_signals = {}
        time = np.linspace(0, self.n_samples / self.sampling_rate, self.n_samples, endpoint=False)
        for freq in self.frequencies:
            signals = [np.sin(2 * np.pi * harmon * freq * time) for harmon in self.harmonics]
            signals += [np.cos(2 * np.pi * harmon * freq * time) for harmon in self.harmonics]
            reference_signals[freq] = np.vstack(signals).T if self.stack_harmonics else np.array(signals)
        return reference_signals

    def _generate_filters(self):
        """
        Generates the filter bank for subband analysis.

        Returns:
            list: A list of filters (b, a) coefficients for each subband.
        """
        filters = []
        nyquist = 0.5 * self.sampling_rate
        subband_width = (40 / nyquist - 6 / nyquist) / self.num_subbands
        for i in range(self.num_subbands):
            band = [6 / nyquist + i * subband_width, 6 / nyquist + (i + 1) * subband_width]
            if band[1] > 1.0:
                band[1] = 1.0
            filters.append(butter(4, band, btype='band'))
        return filters

    def filter_data(self, data):
        """
        Filters the EEG data into multiple subbands.

        Args:
            data (np.ndarray): The EEG data to be filtered.

        Returns:
            np.ndarray: The filtered data for each subband.
        """
        return np.array([filtfilt(b, a, data, axis=-1) for b, a in self.filters])

    def _cca_analysis(self, eeg_data, reference_signal):
        """
        Performs Canonical Correlation Analysis (CCA) between EEG data and reference signals.

        Args:
            eeg_data (np.ndarray): The EEG data to be analyzed.
            reference_signal (np.ndarray): The reference signals to compare against.

        Returns:
            float: The correlation coefficient between the EEG data and reference signals.
        """
        cca = CCA(n_components=1)
        cca.fit([eeg_data.T, reference_signal])
        U, V = cca.transform([eeg_data.T, reference_signal])
        return np.corrcoef(U[:, 0], V[:, 0])[0, 1]

    def __call__(self, eeg_data):
        """
        Performs SSVEP classification based on the chosen method (CCA or FBCCA).

        Args:
            eeg_data (np.ndarray): The EEG data to be analyzed.

        Returns:
            tuple: The detected frequency and the corresponding correlation value.
        """
        max_corr, target_freq = 0, None

        if self.method == 'CCA':
            for freq, ref in self.reference_signals.items():
                if self.stack_harmonics:
                    ref = ref if eeg_data.shape[1] == ref.shape[0] else None
                else:
                    ref = ref if eeg_data.shape[1] == ref.shape[1] else None
                if ref is not None:
                    corr = self._cca_analysis(eeg_data, ref)
                    if corr > max_corr:
                        max_corr, target_freq = corr, freq

        elif self.method == 'FBCCA':
            for freq, ref in self.reference_signals.items():
                corr_sum = 0
                filtered_data = self.filter_data(eeg_data)
                for subband_data in filtered_data:
                    corr = self._cca_analysis(subband_data, ref)
                    corr_sum += corr
                avg_corr = corr_sum / self.num_subbands
                if avg_corr > max_corr:
                    max_corr, target_freq = avg_corr, freq

        return target_freq, max_corr


# Example usage
if __name__ == "__main__":
    frequencies = [9.25, 11.25, 13.25]
    harmonics = np.arange(1, 4)
    sampling_rate = 250
    n_samples = 1025

    # Initialize SSVEP classifier with stacking harmonics for CCA method
    cca_classifier = SSVEPClassifier(frequencies, harmonics, sampling_rate, n_samples, method='CCA', stack_harmonics=True)

    # Example EEG data (randomly generated for illustration purposes, shape: n_channels, n_samples)
    eeg_data = np.random.randn(6, n_samples)  # Assuming 6 channels and 1025 samples

    # Perform CCA analysis
    detected_freq, correlation = cca_classifier(eeg_data)
    print(f"Detected frequency using CCA: {detected_freq} Hz with correlation: {correlation:.3f}")

    # Initialize SSVEP classifier with FBCCA method
    fbcca_classifier = SSVEPClassifier(frequencies, harmonics, sampling_rate, n_samples, method='FBCCA', num_subbands=5)

    # Perform FBCCA analysis
    detected_freq_fbcca, correlation_fbcca = fbcca_classifier(eeg_data)
    print(f"Detected frequency using FBCCA: {detected_freq_fbcca} Hz with correlation: {correlation_fbcca:.3f}")
