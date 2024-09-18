import time
import numpy as np
from brainflow.board_shim import BoardShim

class Segmentation:
    def __init__(self, board, segment_duration, method='continuous'):
        """
        Initializes the Segmentation class.

        Args:
            board: The BoardShim object representing the EEG board.
            segment_duration (float): The duration of each segment in seconds.
            method (str): The method to use for segment retrieval. Options are:
                          'time_wait' for waiting between segment retrievals,
                          'continuous' for continuously fetching new data segments.
        """
        self.board = board
        self.segment_duration = segment_duration
        self.sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        self.n_samples = int(self.sampling_rate * self.segment_duration)
        self.method = method

        # Variables for the 'time_wait' method
        self.last_time = time.time()

        # Variables for the 'continuous' method
        self.prev_data_len = 0

    def get_segment(self):
        """
        Retrieves a segment of data using the chosen method.

        Returns:
            A numpy array representing the data segment, or None if insufficient data is available.
        """
        if self.method == 'time_wait':
            return self._get_segment_time_wait()
        elif self.method == 'continuous':
            return self._get_segment_continuous()
        else:
            raise ValueError("Invalid method specified. Use 'time_wait' or 'continuous'.")

    def _get_segment_time_wait(self):
        """
        Retrieves a segment of data by waiting for segment_duration before fetching the next segment.

        Returns:
            A numpy array representing the data segment, or None if insufficient data is available.
        """
        # Wait until the segment duration has passed
        while time.time() - self.last_time < self.segment_duration:
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting

        # Update the last time to the current time
        self.last_time = time.time()

        # Get the latest segment of data
        data = self.board.get_current_board_data(self.n_samples)
        if data.shape[1] >= self.n_samples:
            segment = data[:, -self.n_samples:]
            return segment
        return None

    def _get_segment_continuous(self):
        """
        Retrieves a new segment of data continuously without waiting between segments.

        Returns:
            A numpy array representing the data segment, or None if insufficient data is available.
        """
        # Get all the data available from the board
        data = self.board.get_board_data()

        # Calculate how much new data is available since the last call
        new_data_len = data.shape[1] - self.prev_data_len

        # Check if there is enough new data for a full segment
        if new_data_len >= self.n_samples:
            # Extract the new segment of data
            segment = data[:, self.prev_data_len:self.prev_data_len + self.n_samples]

            # Update the previous data length to reflect that this data has been consumed
            self.prev_data_len += self.n_samples

            return segment
        else:
            return None

# Example usage
if __name__ == "__main__":
    # Assuming 'board' is a valid BoardShim object already initialized

    # Example using the 'continuous' method
    segmentation_continuous = Segmentation(board, segment_duration=2, method='continuous')
    while True:
        segment = segmentation_continuous.get_segment()
        if segment is not None:
            print("Continuous Segment Retrieved:", segment.shape)
        else:
            print("No segment available, waiting for more data...")
        time.sleep(0.1)  # Small delay to prevent busy-waiting
