
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.ssvep_stim import *

# Example usage
if __name__ == "__main__":
    box_frequencies = [10, 12]
    # box_texts = ['Right', 'Left', 'Up', 'Down']
    # box_text_indices = [0, 1, 3, 2]
    display_index = 1
    display_mode = 'freq'
    monitor_name = 'testMonitor'
    refresh_rate = 60
    
    
    # Using default method: (blocking!) -> use this if running the SSVEPStimulus from a seperate script!
    # start_ssvep_stimulus(box_frequencies = box_frequencies, 
    #                      box_texts = box_texts, 
    #                      box_text_indices = box_text_indices,
    #                      display_index = display_index, 
    #                      display_mode = display_mode, 
    #                      monitor_name = monitor_name)
    
    
    # Start the SSVEP stimulus in a separate process -> use this if running the SSVEPStimulus in the same script as EEG processing or other code
    # Create an instance of the SSVEPStimulusRunner class
    stimulus_process = SSVEPStimulusRunner(box_frequencies = box_frequencies, 
                                            # box_texts = box_texts, 
                                            # box_text_indices = box_text_indices,
                                            display_index = display_index, 
                                            display_mode = display_mode, 
                                            refresh_rate = refresh_rate,
                                            monitor_name = monitor_name)
    # (box_frequencies, box_texts, box_text_indices, display_index, display_mode, monitor_name)
    
    # Start the SSVEP stimulus
    stimulus_process.start()
    
    # Get the actual frequencies
    actual_frequencies = stimulus_process.get_actual_frequencies()
    print("Actual Frequencies:", actual_frequencies)
    
    # Check if the process is still running
    if stimulus_process.is_running():
        print("Process is still running.")
    
    # time.sleep(20)
    
    # Stop the process if needed
    # stimulus_process.stop()
    # print("Process terminated.")