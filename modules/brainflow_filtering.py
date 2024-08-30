import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes

class BF_Filtering:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def bandpass_filter(self, data, lowcut, highcut, order=4):
        """
        Applies a bandpass filter to the data using BrainFlow.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            lowcut (float): The low cut frequency of the filter in Hz.
            highcut (float): The high cut frequency of the filter in Hz.
            order (int, optional): The order of the filter. Default is 4.

        Returns:
            np.ndarray: The bandpass filtered EEG data.

        Raises:
            ValueError: If lowcut or highcut are not provided or are invalid.

        Explanation:
            The 'order' of the filter determines the steepness of the filter's frequency response. 
            A higher order results in a steeper roll-off and more attenuation of frequencies outside 
            the passband. However, it also introduces more phase distortion and computational complexity.
        """
        if lowcut is None or highcut is None:
            raise ValueError("Both lowcut and highcut frequencies must be provided for bandpass filtering.")
        
        DataFilter.perform_bandpass(data, self.sampling_rate, lowcut, highcut, order, FilterTypes.BUTTERWORTH.value, 0)
        return data

    def highpass_filter(self, data, lowcut, order=4):
        """
        Applies a highpass filter to the data using BrainFlow.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            lowcut (float): The low cut frequency of the filter in Hz.
            order (int, optional): The order of the filter. Default is 4.

        Returns:
            np.ndarray: The highpass filtered EEG data.

        Raises:
            ValueError: If lowcut is not provided or is invalid.

        Explanation:
            The 'order' of the filter determines the steepness of the filter's frequency response. 
            A higher order results in a steeper roll-off and more attenuation of frequencies below 
            the cutoff frequency. However, it also introduces more phase distortion and computational complexity.
        """
        if lowcut is None:
            raise ValueError("Lowcut frequency must be provided for highpass filtering.")
        
        DataFilter.perform_highpass(data, self.sampling_rate, lowcut, order, FilterTypes.BUTTERWORTH.value, 0)
        return data

    def lowpass_filter(self, data, highcut, order=4):
        """
        Applies a lowpass filter to the data using BrainFlow.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            highcut (float): The high cut frequency of the filter in Hz.
            order (int, optional): The order of the filter. Default is 4.

        Returns:
            np.ndarray: The lowpass filtered EEG data.

        Raises:
            ValueError: If highcut is not provided or is invalid.

        Explanation:
            The 'order' of the filter determines the steepness of the filter's frequency response. 
            A higher order results in a steeper roll-off and more attenuation of frequencies above 
            the cutoff frequency. However, it also introduces more phase distortion and computational complexity.
        """
        if highcut is None:
            raise ValueError("Highcut frequency must be provided for lowpass filtering.")
        
        DataFilter.perform_lowpass(data, self.sampling_rate, highcut, order, FilterTypes.BUTTERWORTH.value, 0)
        return data

    def notch_filter(self, data, notch_freq, quality_factor=30.0):
        """
        Applies a notch filter to the data to remove power line noise using BrainFlow.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            notch_freq (float): The frequency to be removed from the data (e.g., 50 Hz or 60 Hz).
            quality_factor (float, optional): Quality factor for the notch filter. Default is 30.0.

        Returns:
            np.ndarray: The notch filtered EEG data.

        Raises:
            ValueError: If notch_freq is not provided or is invalid.
        """
        if notch_freq is None:
            raise ValueError("Notch frequency must be provided for notch filtering.")
        
        DataFilter.perform_bandstop(data, self.sampling_rate, notch_freq - 0.5, notch_freq + 0.5, 2, FilterTypes.BUTTERWORTH.value, 0)
        return data

    def bandstop_filter(self, data, lowcut, highcut, order=4):
        """
        Applies a bandstop filter to the data using BrainFlow.

        Args:
            data (np.ndarray): The EEG data for a single channel.
            lowcut (float): The low cut frequency of the filter in Hz.
            highcut (float): The high cut frequency of the filter in Hz.
            order (int, optional): The order of the filter. Default is 4.

        Returns:
            np.ndarray: The bandstop filtered EEG data.

        Raises:
            ValueError: If lowcut or highcut are not provided or are invalid.

        Explanation:
            The 'order' of the filter determines the steepness of the filter's frequency response. 
            A higher order results in a steeper roll-off and more attenuation of frequencies within 
            the stopband. However, it also introduces more phase distortion and computational complexity.
        """
        if lowcut is None or highcut is None:
            raise ValueError("Both lowcut and highcut frequencies must be provided for bandstop filtering.")
        
        DataFilter.perform_bandstop(data, self.sampling_rate, lowcut, highcut, order, FilterTypes.BUTTERWORTH.value, 0)
        return data

    def filter_data(self, data, filter_type="bandpass", **kwargs):
        """
        Applies the specified filter to the EEG data using BrainFlow.

        Args:
            data (np.ndarray): The EEG data to be filtered.
            filter_type (str): The type of filter to apply. Options are "bandpass", "highpass", "lowpass", "notch", "bandstop".
            kwargs: Additional arguments for the filters, such as 'lowcut', 'highcut', 'order', 'notch_freq', and 'quality_factor'.

        Returns:
            np.ndarray: The filtered EEG data.

        Raises:
            ValueError: If required filter parameters are missing or invalid.
        """
        if filter_type == "bandpass":
            return np.apply_along_axis(self.bandpass_filter, 1, data, 
                                       kwargs["lowcut"], 
                                       kwargs["highcut"], 
                                       kwargs.get("order", 4))
        elif filter_type == "highpass":
            return np.apply_along_axis(self.highpass_filter, 1, data, 
                                       kwargs["lowcut"], 
                                       kwargs.get("order", 4))
        elif filter_type == "lowpass":
            return np.apply_along_axis(self.lowpass_filter, 1, data, 
                                       kwargs["highcut"], 
                                       kwargs.get("order", 4))
        elif filter_type == "notch":
            return np.apply_along_axis(self.notch_filter, 1, data, 
                                       kwargs["notch_freq"], 
                                       kwargs.get("quality_factor", 30.0))
        elif filter_type == "bandstop":
            return np.apply_along_axis(self.bandstop_filter, 1, data, 
                                       kwargs["lowcut"], 
                                       kwargs["highcut"], 
                                       kwargs.get("order", 4))
        else:
            raise ValueError("Invalid filter type. Options are 'bandpass', 'highpass', 'lowpass', 'notch', 'bandstop'.")


if __name__ == "__main__":
    # Assuming the BF_Filtering class is defined as provided

    # Create an instance of the BF_Filtering class with a sampling rate of 250 Hz
    sampling_rate = 250
    filtering = BF_Filtering(sampling_rate)

    # Generate some example EEG data (e.g., 8 channels, 1000 samples)
    eeg_data = np.random.randn(8, 1000)

    # Apply a bandpass filter to the EEG data
    lowcut = 1.0  # Low cut frequency in Hz
    highcut = 50.0  # High cut frequency in Hz
    filtered_data_bandpass = filtering.filter_data(eeg_data, filter_type="bandpass", lowcut=lowcut, highcut=highcut)

    # Apply a highpass filter to the EEG data
    lowcut = 1.0  # Low cut frequency in Hz
    filtered_data_highpass = filtering.filter_data(eeg_data, filter_type="highpass", lowcut=lowcut)

    # Apply a lowpass filter to the EEG data
    highcut = 50.0  # High cut frequency in Hz
    filtered_data_lowpass = filtering.filter_data(eeg_data, filter_type="lowpass", highcut=highcut)

    # Apply a notch filter to the EEG data to remove 50 Hz power line noise
    notch_freq = 50.0  # Notch frequency in Hz
    filtered_data_notch = filtering.filter_data(eeg_data, filter_type="notch", notch_freq=notch_freq)

    # Apply a bandstop filter to the EEG data
    lowcut = 48.0  # Low cut frequency in Hz
    highcut = 52.0  # High cut frequency in Hz
    filtered_data_bandstop = filtering.filter_data(eeg_data, filter_type="bandstop", lowcut=lowcut, highcut=highcut)
    
    # Print the shapes of the filtered data to verify the filtering process
    print("Original Data (first 10 samples of channel 0):", eeg_data[0, :10])
    print("Bandpass Filtered Data (first 10 samples of channel 0):", filtered_data_bandpass[0, :10])
    print("Highpass Filtered Data (first 10 samples of channel 0):", filtered_data_highpass[0, :10])
    print("Lowpass Filtered Data (first 10 samples of channel 0):", filtered_data_lowpass[0, :10])
    print("Notch Filtered Data (first 10 samples of channel 0):", filtered_data_notch[0, :10])
    print("Bandstop Filtered Data (first 10 samples of channel 0):", filtered_data_bandstop[0, :10])