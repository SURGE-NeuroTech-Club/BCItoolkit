import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError

class BrainFlowBoardSetup:
    """
    A class to manage the setup and control of a BrainFlow board.
    Also allows for use of all BoardShim attributes (even if not explicity defined in this class) 
        --> https://brainflow.readthedocs.io/en/stable/UserAPI.html#brainflow-board-shim

    Attributes:
        board_id (int): The ID of the BrainFlow board.
        serial_port (str): The serial port to which the BrainFlow board is connected.
        params (BrainFlowInputParams): An instance of the BrainFlowInputParams class, representing the input parameters for the BrainFlow board.
        board (BoardShim): An instance of the BoardShim class, representing the BrainFlow board.

    Methods:
        setup(): Prepares the session and starts the data stream from the BrainFlow board.
        stop(): Stops the data stream and releases the session of the BrainFlow board.
        get_board_data(): Retrieves the current data from the BrainFlow board.
        show_params(): Prints the current parameters of the BrainFlowInputParams instance.
    """

    def __init__(self, board_id, serial_port, **kwargs):
        """
        Initializes the BrainFlowBoardSetup class with the given board ID, serial port, and additional parameters.

        Args:
            board_id (int): The ID of the BrainFlow board.
            serial_port (str): The serial port to which the BrainFlow board is connected.
            **kwargs: Additional keyword arguments to be set as attributes on the BrainFlowInputParams instance.
        """
        self.board_id = board_id
        self.serial_port = serial_port
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

    def setup(self):
        """
        Prepares the session and starts the data stream from the BrainFlow board.

        This method initializes the BoardShim with the given board ID and input parameters,
        prepares the session, and starts the data stream. If an error occurs, it prints the error message
        and sets the board attribute to None.
        """
        self.board = BoardShim(self.board_id, self.params)
        try:
            self.board.prepare_session()
            self.session_prepared = True
            self.board.start_stream(450000)
            self.streaming = True  # Flag to indicate if streaming is active
            print("Board setup and streaming started successfully")
        except BrainFlowError as e:
            print(f"Error setting up board: {e}")
            self.board = None
    
    def show_params(self):
        """
        Prints the current parameters of the BrainFlowInputParams instance.
        """
        print("Current BrainFlow Input Parameters:")
        for key, value in vars(self.params).items():
            print(f"{key}: {value}")

    def get_board_data(self):
        """
        Retrieves the current data from the BrainFlow board.

        Returns:
            numpy.ndarray: The current data from the BrainFlow board if the board is set up,
            otherwise None.
        """
        if self.board is not None:
            return self.board.get_board_data()
        else:
            print("Board is not set up")
            return None
    
    def stop(self):
        """
        Stops the data stream and releases the session of the BrainFlow board.

        This method stops the data stream and releases the session. If an error occurs,
        it prints the error message, unless the error is "BOARD_NOT_CREATED_ERROR:15 unable to stop streaming session".
        """
        try:
            if self.board is not None:
                if self.streaming:
                    self.board.stop_stream()
                    self.streaming = False
                    print("\nStreaming stopped")
                if self.session_prepared:
                    self.board.release_session()
                    self.session_prepared = False
                    print("Session released")
        except BrainFlowError as e:
            if "BOARD_NOT_CREATED_ERROR:15" not in str(e):
                print(f"Error stopping board: {e}")  
                                
    def __getattr__(self, name):
        """
        Delegates attribute access to the BoardShim instance if the attribute is not found in the current instance.

        Args:
            name (str): The name of the attribute.

        Returns:
            The attribute from the BoardShim instance if it exists, otherwise raises AttributeError.
        """
        if self.board is not None and hasattr(self.board, name):
            return getattr(self.board, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __del__(self):
        """
        Ensures that the data stream is stopped and the session is released when the object is deleted.

        This method is called when the object is about to be destroyed. It stops the data stream
        and releases the session if the board is set up.
        """
        self.stop()
