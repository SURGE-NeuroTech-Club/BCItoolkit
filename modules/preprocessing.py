import time
import numpy as np
from brainflow.board_shim import BoardShim
from scipy.signal import butter, lfilter

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

    # def filter_data(self, data, lowcut=0.5, highcut=30.0, order=5):
    #     """
    #     Applies a bandpass filter to the EEG data using BrainFlow library.

    #     Args:
    #         data (np.ndarray): The EEG data to be filtered.
    #         lowcut (float): The low cut frequency in Hz.
    #         highcut (float): The high cut frequency in Hz.
    #         order (int): The order of the filter.

    #     Returns:
    #         np.ndarray: The filtered EEG data.
    #     """
    #     for channel in range(data.shape[0]):
    #         DataFilter.perform_bandpass(data[channel], self.sampling_rate, lowcut, highcut, order, FilterTypes.BUTTERWORTH, 0)
    #     return data

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
