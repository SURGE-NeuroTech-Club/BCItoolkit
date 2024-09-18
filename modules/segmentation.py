import time
import numpy as np
from brainflow.board_shim import BoardShim

class Segmentation:
    def __init__(self, board, segment_duration):
        """
        Initializes the Segmentation class.

        Args:
            board: The BoardShim object representing the EEG board.
            segment_duration (float): The duration of each segment in seconds.
        """
        self.board = board
        self.segment_duration = segment_duration
        self.sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        self.n_samples = int(self.sampling_rate * self.segment_duration)
        self.last_time = time.time()

    def get_segment(self):
        """
        Retrieves the most recent segment of data from the board.

        This method fetches the latest segment of data based on the segment duration. 
        It uses the get_current_board_data function from the BoardShim library to retrieve 
        the latest samples available on the board.

        Returns:
            A numpy array representing the data segment, or None if insufficient data is available.
        """
        data = self.board.get_current_board_data(self.n_samples)
        if data.shape[1] >= self.n_samples:
            segment = data[:, -self.n_samples:]
            return segment
        return None

    def get_segment_time(self):
        """
        Retrieves a segment of data by waiting for segment_duration before fetching the next segment.

        This method ensures that each segment retrieved is spaced out by the specified segment duration.
        It waits until the appropriate amount of time has passed since the last segment was retrieved 
        before fetching the next one.

        Returns:
            A numpy array representing the data segment, or None if insufficient data is available.
        """
        # Wait until the segment duration has passed
        while time.time() - self.last_time < self.segment_duration:
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting

        # Update the last time to the current time
        self.last_time = time.time()

        # Get the latest segment of data
        return self.get_segment()

if __name__ == "__main__":
    # Assuming 'board' is a valid BoardShim object already initialized

    # Example using the 'time_wait' method
    segmentation = Segmentation(board, segment_duration=2)
    while True:
        segment = segmentation.get_segment_time_wait()
        if segment is not None:
            print("Segment Retrieved:", segment.shape)
        else:
            print("Insufficient data, waiting...")


############
# Old method: Segment duration control has to be done outside of the class - this code will not wait for <segment_duration> to return new data, means potential for overlapping data between segments.
############

# import numpy as np
# from brainflow.board_shim import BoardShim

# class Segmentation:
#     def __init__(self, board, segment_duration):
#         self.board = board
#         self.segment_duration = segment_duration
#         self.sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
#         self.n_samples = int(self.sampling_rate * self.segment_duration)

#     def get_segment(self):
#         data = self.board.get_current_board_data(self.n_samples)
#         if data.shape[1] >= self.n_samples:
#             segment = data[:, -self.n_samples:]
#             return segment
#         return None
