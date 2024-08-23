# from brainflow_board_setup import BrainFlowBoardSetup

# Example usage of the BrainFlowBoardSetup class
    # https://brainflow.readthedocs.io/en/stable/UserAPI.html#brainflow-board-shim

def main():
    # Define board ID and serial port (replace with your actual values)
    board_id = 0  # Example: 0 for synthetic board
    serial_port = "COM3"  # Replace with your actual serial port
    
    # Additional BrainFlow input parameters (if any)
    other_params = {
        "timeout": 15,  # Example of an additional parameter
    }

    # Initialize the BrainFlowBoardSetup class
    board_setup = BrainFlowBoardSetup(board_id, serial_port, **other_params)

    # Show the current input parameters
    board_setup.show_params()

    # Set up the board and start streaming data
    board_setup.setup()

    # Retrieve data from the board (e.g., after some time)
    import time
    time.sleep(5)  # Allow some time to gather data
    
    data = board_setup.get_board_data()
    if data is not None:
        print("Retrieved data shape:", data.shape)
        print(data)  # Display the data (or process it as needed)

    # Stop the board and release the session
    board_setup.stop()

if __name__ == "__main__":
    main()
