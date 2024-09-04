import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BoardIds
import serial.tools.list_ports

class BrainFlowBoardSetup:
    """
    A class to manage the setup and control of a BrainFlow board.
    Also allows for use of all BoardShim attributes (even if not explicitly defined in this class).
    
    Attributes:
        name (str): A user-friendly name or identifier for the board setup instance.
        board_id (int): The ID of the BrainFlow board to use.
        serial_port (str): The serial port to which the BrainFlow board is connected.
        params (BrainFlowInputParams): Instance of BrainFlowInputParams representing the board's input parameters.
        board (BoardShim): Instance of BoardShim representing the active board.
        session_prepared (bool): Flag indicating if the session has been prepared.
        streaming (bool): Flag indicating if the board is actively streaming data.
        eeg_channels (list): List of EEG channel indices for the board.
        sampling_rate (int): Sampling rate of the board.
    """

    _id_counter = 0  # Class-level variable to assign default IDs

    def __init__(self, board_id, serial_port=None, name=None, **kwargs):
        """
        Initializes the BrainFlowBoardSetup class with the given board ID, serial port, and additional parameters.

        Args:
            board_id (int): The ID of the BrainFlow board.
            serial_port (str): The serial port to which the BrainFlow board is connected.
            name (str): A user-friendly name or identifier for this instance. Defaults to 'Board X'.
            **kwargs: Additional keyword arguments to be set as attributes on the BrainFlowInputParams instance.
        """
        self.board_id = board_id
        self.serial_port = serial_port

        # Assign default name if not provided, based on the class-level ID counter
        self.name = name or f"Board {BrainFlowBoardSetup._id_counter}"
        BrainFlowBoardSetup._id_counter += 1

        self.params = BrainFlowInputParams()
        self.params.serial_port = self.serial_port
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        
        # Set additional parameters if provided
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                print(f"Warning: {key} is not a valid parameter for BrainFlowInputParams")

        self.board = None
        self.session_prepared = False
        self.streaming = False

    def find_device_ports(self):
        """
        Finds all compatible BrainFlow devices by checking the available serial ports.

        This method iterates over available serial ports on the computer and attempts
        to detect and verify BrainFlow-compatible devices by initializing a session.

        Returns:
            list: A list of dictionaries containing 'port', 'serial_number', and 'description' for each compatible device.
                  Returns an empty list if no devices are found.
        """
        # Suppress logs from BrainFlow using internal logging level control
        BoardShim.disable_board_logger()  # Disable all logs

        ports = serial.tools.list_ports.comports()
        compatible_ports = []

        for port in ports:
            try:
                self.params.serial_port = port.device
                board = BoardShim(self.board_id, self.params)
                board.prepare_session()
                board.release_session()
                
                # Collect information about the compatible device
                device_info = {
                    'port': port.device,
                    'serial_number': port.serial_number,
                    'description': port.description
                }
                print(f"Compatible device found: Serial Number: {port.serial_number}, Description: {port.description}")
                compatible_ports.append(device_info)
            except BrainFlowError:
                continue
        
        if not compatible_ports:
            print(f"No compatible BrainFlow devices found.")
        
        # Re-enable logging after checking ports
        BoardShim.enable_board_logger()

        return compatible_ports
    
    def setup(self):
        """
        Prepares the session and starts the data stream from the BrainFlow board.

        If no serial port is provided during initialization, this method attempts to auto-detect
        a compatible device. Once the board is detected or provided, it prepares the session and starts streaming.

        Raises:
            BrainFlowError: If the board fails to prepare the session or start streaming.
        """
        if self.serial_port is None:
            print(f"No serial port provided, attempting to auto-detect...")
            ports_info = self.find_device_ports()
            if ports_info:
                self.serial_port = ports_info[0]['port']  # Default to the first detected port
            else:
                print(f"No compatible device found. Setup failed.")
                return

        self.params.serial_port = self.serial_port
        self.board = BoardShim(self.board_id, self.params)
        try:
            self.board.prepare_session()
            self.session_prepared = True
            self.board.start_stream(450000)
            self.streaming = True  # Flag to indicate if streaming is active
            print(f"[{self.name}, {self.serial_port}] Board setup and streaming started successfully.")
        except BrainFlowError as e:
            print(f"[{self.name}, {self.serial_port}] Error setting up board: {e}")
            self.board = None
    
    def show_params(self):
        """
        Prints the current parameters of the BrainFlowInputParams instance.

        This method provides a simple way to inspect the current input parameters
        being used to configure the BrainFlow board.
        """
        print(f"[{self.name}] Current BrainFlow Input Parameters:")
        for key, value in vars(self.params).items():
            print(f"{key}: {value}")

    def get_board_data(self):
        """
        Retrieves the current data from the BrainFlow board.

        This method fetches the most recent data collected from the board, including EEG, accelerometer, and other sensor data.

        Returns:
            numpy.ndarray: The current data from the BrainFlow board if the board is set up.
            None: If the board is not set up.
        """
        if self.board is not None:
            return self.board.get_board_data()
        else:
            print(f"[{self.name}] Board is not set up.")
            return None
        
    def insert_marker(self, marker, verbose=True):
        """
        Inserts a marker into the data stream at the current time.

        This is useful for marking specific events in the data stream for later analysis.

        Args:
            marker (float): The marker value to be inserted.
            verbose (bool): Whether to print a confirmation message. Default is True.
        """
        self.verbose = verbose

        if self.board is not None and self.streaming:
            try:
                self.board.insert_marker(marker)
                if self.verbose:
                    print(f"[{self.name}] Marker {marker} inserted successfully.")
            except BrainFlowError as e:
                print(f"[{self.name}] Error inserting marker: {e}")
        else:
            print(f"[{self.name}] Board is not streaming, cannot insert marker.")
    
    def stop(self):
        """
        Stops the data stream and releases the session of the BrainFlow board.

        This method safely stops the data stream and releases any resources used by the BrainFlow board.
        It also resets the streaming and session flags.
        """
        try:
            if self.board is not None:
                if self.streaming:
                    self.board.stop_stream()
                    self.streaming = False
                    print(f"[{self.name}, {self.serial_port}] Streaming stopped.")
                if self.session_prepared:
                    self.board.release_session()
                    self.session_prepared = False
                    print(f"[{self.name}, {self.serial_port}] Session released.")
        except BrainFlowError as e:
            if "BOARD_NOT_CREATED_ERROR:15" not in str(e):
                print(f"[{self.name}, {self.serial_port}] Error stopping board: {e}")

    def __getattr__(self, name):
        """
        Delegates attribute access to the BoardShim instance if the attribute is not found in the current instance.

        This method allows access to BoardShim-specific attributes that may not be defined directly
        in the BrainFlowBoardSetup class.

        Args:
            name (str): The name of the attribute to be accessed.

        Returns:
            The attribute from the BoardShim instance if it exists.
        
        Raises:
            AttributeError: If the attribute is not found in the current instance or the BoardShim instance.
        """
        if self.board is not None and hasattr(self.board, name):
            return getattr(self.board, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __del__(self):
        """
        Ensures that the data stream is stopped and the session is released when the object is deleted.

        This method ensures that the session is properly released and resources are freed when the object is garbage collected.
        """
        self.stop()

#######
# Example streaming from a single board
######
if __name__ == "__main__":
    import time

    board_id_cyton = BoardIds.CYTON_BOARD.value

    brainflow_board = BrainFlowBoardSetup(board_id=board_id_cyton)

    brainflow_board.setup()

    # Stream from both boards for 5 seconds
    time.sleep(5)

    # Retrieve and print data from the second board
    data = brainflow_board.get_board_data()
    print(f"Data from brainflow_board: {data}")

    # Stop the streaming and release the sessions for both boards
    brainflow_board.stop()


############
# Example streaming from two boards simultaneously
###########
## Method 1 - automatically finding multiple devices and starting them
# if __name__ == "__main__":
#     import time

#     board_id_cyton = BoardIds.CYTON_BOARD.value

#     # Instantiate BrainFlowBoardSetup for the first board with verbose=False to suppress logs
#     brainflow_setup_1 = BrainFlowBoardSetup(board_id=board_id_cyton)

#     # Instantiate BrainFlowBoardSetup for the second board with verbose=True for detailed output
#     brainflow_setup_2 = BrainFlowBoardSetup(board_id=board_id_cyton)

#     # Set up the first board and start streaming
#     brainflow_setup_1.setup()

#     # Set up the second board and start streaming (with detailed logs)
#     brainflow_setup_2.setup()

#     # Stream for 5 seconds
#     time.sleep(5)

#     # Retrieve and print data from the first board
#     data_1 = brainflow_setup_1.get_board_data()
#     print(f"Data from board 1: {data_1}")

#     # Retrieve and print data from the second board
#     data_2 = brainflow_setup_2.get_board_data()
#     print(f"Data from board 2: {data_2}")

#     # Stop streaming and release both boards
#     brainflow_setup_1.stop()
#     brainflow_setup_2.stop()

## Method 2 - Detecting compatible devices and assigning them consistently -> ideal for ensuring the same board is assigned to the same port on each run
# if __name__ == "__main__":
#     import time

#     board_id_cyton = BoardIds.CYTON_BOARD.value

#     # Instantiate BrainFlowBoardSetup with verbose enabled for detailed output
#     brainflow_setup_detector = BrainFlowBoardSetup(board_id=board_id_cyton)

#     # Find all compatible devices
#     compatible_ports = brainflow_setup_detector.find_device_ports()

#     # Check if at least two devices are found
#     if len(compatible_ports) >= 2:
#         serial_port_1 = compatible_ports[0]['port']
#         serial_port_2 = compatible_ports[1]['port']
        
#         # Instantiate BrainFlowBoardSetup for the first board
#         brainflow_setup_1 = BrainFlowBoardSetup(board_id=board_id_cyton, serial_port=serial_port_1)

#         # Instantiate BrainFlowBoardSetup for the second board
#         brainflow_setup_2 = BrainFlowBoardSetup(board_id=board_id_cyton, serial_port=serial_port_2)

#         # Set up the first board and start streaming
#         brainflow_setup_1.setup()

#         # Set up the second board and start streaming
#         brainflow_setup_2.setup()

#         # Stream from both boards for 5 seconds
#         time.sleep(5)

#         # Retrieve and print data from the first board
#         data_1 = brainflow_setup_1.get_board_data()
#         print(f"Data from board 1: {data_1}")

#         # Retrieve and print data from the second board
#         data_2 = brainflow_setup_2.get_board_data()
#         print(f"Data from board 2: {data_2}")

#         # Stop the streaming and release the sessions for both boards
#         brainflow_setup_1.stop()
#         brainflow_setup_2.stop()

#     else:
#         print("Not enough compatible devices found.")
