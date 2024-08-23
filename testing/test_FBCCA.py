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
        for i in range(self.num_subbands):
            band = [low * (i + 1), high * (i + 1)]
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

## Main script integration
def main():
    params = BrainFlowInputParams()
    board_id = BoardIds.UNICORN_BOARD
    frequencies = [9.25, 11.25, 13.25, 15.25]
    button_pos = [0, 2, 3, 1]
    segment_duration = 5
    display = 1
    harmonics = np.arange(1, 6)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    n_samples = sampling_rate * segment_duration
    eeg_channels = BoardShim.get_eeg_names(board_id)
    channel_names = ["O1", "O2", "Oz", "Pz", "P3", "P4", "POz", "P1"]
    channel_mapping = dict(zip(eeg_channels, channel_names))

    print(f"Sampling Rate: {sampling_rate}")
    print(f"Default Channels: {eeg_channels}")
    print(f"Channel Mapping: {channel_mapping}")

    def run_stimulus():
        stimulus = SSVEPStimulus(frequencies, box_text_indices=button_pos, show_both=True, display_index=display)
        stimulus.run()

    key_listener = KeyListener()
    key_listener.run_listener()
    board = BoardShim(BoardIds.UNICORN_BOARD, params)
    board.setup()
    print(f"....Warming up....")
    time.sleep(5)
    segmenter = PreProcess(board, segment_duration=segment_duration)
    classifier = ClassifySSVEP(frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=False)
    classifier_stacked = ClassifySSVEP(frequencies, harmonics, sampling_rate, n_samples, stack_harmonics=True)
    fbcca_classifier = FBCCA(frequencies, harmonics, sampling_rate, n_samples)

    stimulus_thread = threading.Thread(target=run_stimulus)
    stimulus_thread.start()

    try:
        while not key_listener.stop_flag:
            segment = segmenter.get_segment()
            if segment is not None:
                eeg_segment = segment[:8, :]
                filtered_segment = segmenter.filter_data(eeg_segment)
                print("Filtered data shape:", filtered_segment.shape)
                detected_freq, correlation = classifier.cca_analysis(filtered_segment)
                detected_freq_stacked, correlation_stacked = classifier_stacked.cca_analysis(filtered_segment)
                detected_freq_fbcca, correlation_fbcca = fbcca_classifier.fbcca_analysis(filtered_segment)
                print(f"CCA: Detected frequency: {detected_freq} Hz with correlation: {correlation}")
                print(f"Stacked CCA: Detected frequency: {detected_freq_stacked} Hz with correlation: {correlation_stacked}")
                print(f"FBCCA: Detected frequency: {detected_freq_fbcca} Hz with correlation: {correlation_fbcca}")

            time.sleep(segmenter.segment_duration)
    except KeyboardInterrupt:
        pass
    finally:
        board.stop()
        print("\nSession Exited Successfully\n")

if __name__ == "__main__":
    main()
