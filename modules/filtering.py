import numpy as np
from scipy.signal import butter, lfilter, iirnotch, filtfilt

class Filtering:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def bandpass_filter(self, data, lowcut, highcut, order=6):
        """
        Applies a bandpass filter to the data.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            lowcut (float): The low cut frequency of the filter in Hz.
            highcut (float): The high cut frequency of the filter in Hz.
            order (int): The order of the filter.

        Returns:
            np.ndarray: The bandpass filtered EEG data.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def highpass_filter(self, data, lowcut, order=5):
        """
        Applies a highpass filter to the data.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            lowcut (float): The low cut frequency of the filter in Hz.
            order (int): The order of the filter.

        Returns:
            np.ndarray: The highpass filtered EEG data.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        b, a = butter(order, low, btype='high')
        y = filtfilt(b, a, data)
        return y

    def lowpass_filter(self, data, highcut, order=5):
        """
        Applies a lowpass filter to the data.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            highcut (float): The high cut frequency of the filter in Hz.
            order (int): The order of the filter.

        Returns:
            np.ndarray: The lowpass filtered EEG data.
        """
        nyquist = 0.5 * self.sampling_rate
        high = highcut / nyquist
        b, a = butter(order, high, btype='low')
        y = filtfilt(b, a, data)
        return y

    def notch_filter(self, data, notch_freq, quality_factor=30.0):
        """
        Applies a notch filter to the data to remove power line noise.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            notch_freq (float): The frequency to be removed from the data (e.g., 50 Hz or 60 Hz).
            quality_factor (float): Quality factor for the notch filter. Default is 30.0.

        Returns:
            np.ndarray: The notch filtered EEG data.
        """
        nyquist = 0.5 * self.sampling_rate
        w0 = notch_freq / nyquist
        b, a = iirnotch(w0, quality_factor)
        y = filtfilt(b, a, data)
        return y

    def bandstop_filter(self, data, lowcut, highcut, order=5):
        """
        Applies a bandstop filter to the data.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            lowcut (float): The low cut frequency of the filter in Hz.
            highcut (float): The high cut frequency of the filter in Hz.
            order (int): The order of the filter.

        Returns:
            np.ndarray: The bandstop filtered EEG data.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='bandstop')
        y = filtfilt(b, a, data)
        return y

    def filter_data(self, data, filter_type="bandpass", **kwargs):
        """
        Applies the specified filter to the EEG data.

        Args:
            data (np.ndarray): The EEG data to be filtered.
            filter_type (str): The type of filter to apply. Options are "bandpass", "highpass", "lowpass", "notch", "bandstop".
            kwargs: Additional arguments for the filters, such as 'lowcut', 'highcut', 'order', 'notch_freq', and 'quality_factor'.

        Returns:
            np.ndarray: The filtered EEG data.
        """
        if filter_type == "bandpass":
            return np.apply_along_axis(self.bandpass_filter, 1, data, 
                                       kwargs.get("lowcut", 0.5), 
                                       kwargs.get("highcut", 30.0), 
                                       kwargs.get("order", 5))
        elif filter_type == "highpass":
            return np.apply_along_axis(self.highpass_filter, 1, data, 
                                       kwargs.get("lowcut", 0.5), 
                                       kwargs.get("order", 5))
        elif filter_type == "lowpass":
            return np.apply_along_axis(self.lowpass_filter, 1, data, 
                                       kwargs.get("highcut", 30.0), 
                                       kwargs.get("order", 5))
        elif filter_type == "notch":
            return np.apply_along_axis(self.notch_filter, 1, data, 
                                       kwargs.get("notch_freq", 50.0), 
                                       kwargs.get("quality_factor", 30.0))
        elif filter_type == "bandstop":
            return np.apply_along_axis(self.bandstop_filter, 1, data, 
                                       kwargs.get("lowcut", 48.0), 
                                       kwargs.get("highcut", 52.0), 
                                       kwargs.get("order", 5))
        else:
            raise ValueError("Invalid filter type. Options are 'bandpass', 'highpass', 'lowpass', 'notch', 'bandstop'.")
