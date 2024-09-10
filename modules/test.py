from brainflow.board_shim import BoardShim, BrainFlowInputParams

# Cyton board ID (replace with your specific board ID if different)
CYTON_BOARD_ID = 0

# Optionally enable the BrainFlow logger for debugging purposes
BoardShim.enable_dev_board_logger()

# Get the EEG channels for the Cyton board using the class method
eeg_channels = BoardShim.get_eeg_channels(CYTON_BOARD_ID)

# Get additional channel info (e.g., EXG channels if available)
exg_channels = BoardShim.get_exg_channels(CYTON_BOARD_ID)

# Print the EEG and EXG channels
print("EEG Channels for Cyton Board:", eeg_channels)
print("EXG Channels (for external electrodes):", exg_channels)

# Assuming the Cyton board uses SRB2 as a reference for all channels and a hardware DRL for ground
reference_channel = 'SRB2'  # Built-in reference, part of the board's internal design
ground_channel = 'DRL'      # Hardware-based ground

print(f"Reference Channel: {reference_channel}")
print(f"Ground Channel: {ground_channel}")
