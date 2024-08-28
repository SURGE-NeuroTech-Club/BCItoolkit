import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import time
import numpy as np
from psychopy import visual, event, core, monitors
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from mvlearn.embed import CCA

# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import BrainFlowBoardSetup
from modules.filtering import *
from modules.segmentation import Segmentation
from modules.classification import SSVEPClassifier

class SSVEPStimulus:
    def __init__(self, box_frequencies, box_texts=None, box_text_indices=None, display_index=0, monitor_name='testMonitor'):
        self.box_frequencies = box_frequencies
        self.box_texts = box_texts
        self.box_text_indices = box_text_indices

        # Setup PsychoPy window and monitor
        monitor = monitors.Monitor(monitor_name)
        self.win = visual.Window(
            monitor=monitor,
            screen=display_index,
            fullscr=False,  # Set to False for easier debugging; change to True for full screen
            color='black',
            units='pix',
            allowGUI=False,
            winType='pyglet',
            autoLog=False
        )

        self.refresh_rate = self.win.getActualFrameRate() or 60  # Use default if unable to measure
        print(f"Measured Refresh Rate: {self.refresh_rate:.2f} Hz")

        # Calculate actual frequencies
        self.actual_frequencies = self.calculate_actual_frequencies(box_frequencies)
        self.boxes = self.create_boxes()

        self.is_running = False
        self.frame_count = 0

    def calculate_actual_frequencies(self, desired_frequencies):
        actual_frequencies = []
        for freq in desired_frequencies:
            frames_per_cycle = round(self.refresh_rate / freq)
            actual_freq = self.refresh_rate / frames_per_cycle
            actual_frequencies.append(actual_freq)
        return actual_frequencies

    def create_boxes(self):
        boxes = []
        radius = min(self.win.size) // 3
        centerX, centerY = 0, 0
        num_boxes = len(self.actual_frequencies)

        for i, freq in enumerate(self.actual_frequencies):
            angle = 2 * np.pi * i / num_boxes
            pos = (centerX + int(radius * np.cos(angle)), centerY + int(radius * np.sin(angle)))
            box = visual.Rect(win=self.win, width=150, height=150, fillColor='white', pos=pos)
            boxes.append({
                "box": box,
                "frequency": freq,
                "on": True,
                "frame_count": 0
            })
        return boxes

    def start(self):
        self.is_running = True
        self.run_loop()

    def run_loop(self):
        while self.is_running:
            self.update()
            time.sleep(1.0 / self.refresh_rate)  # Maintain the refresh rate consistency

    def update(self):
        self.frame_count += 1
        for box_info in self.boxes:
            flicker_period = self.refresh_rate / box_info["frequency"]
            if (self.frame_count % flicker_period) < (flicker_period / 2):
                if not box_info["on"]:
                    box_info["box"].setAutoDraw(True)
                    box_info["on"] = True
            else:
                if box_info["on"]:
                    box_info["box"].setAutoDraw(False)
                    box_info["on"] = False

        self.win.flip()
        if 'escape' in event.getKeys():
            self.stop()

    def stop(self):
        self.is_running = False
        self.win.close()
        core.quit()

def run_processing(board, stimulus, harmonics, sampling_rate, n_samples):
    segmentation_time_wait = Segmentation(board, segment_duration=2)
    actual_freqs = stimulus.actual_frequencies

    cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
                                     harmonics=harmonics, 
                                     sampling_rate=sampling_rate, 
                                     n_samples=n_samples, 
                                     method='CCA', 
                                     stack_harmonics=True)

    while not stimulus.is_finished():
        segment = segmentation_time_wait.get_segment_time()
        if segment is not None:
            print("Segment Retrieved:", segment.shape)
            eeg_segment = segment[:8, :] 

            detected_freq, correlation = cca_classifier(eeg_segment)
            print(f"Detected frequency using CCA: {detected_freq} Hz with correlation: {correlation:.3f}")

def main():
    # Setup parameters
    serial_port = 'COM4'
    board_id = BoardIds.SYNTHETIC_BOARD
    frequencies = [9.25, 11.25, 13.25, 15.25]
    buttons = ['Right', 'Left', 'Up', 'Down']
    button_pos = [0, 2, 3, 1]
    segment_duration = 4
    display = 0

    harmonics = np.arange(1, 4)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    n_samples = sampling_rate * segment_duration 

    eeg_channels = BoardShim.get_eeg_names(board_id)
    channel_names = ["O1", "O2", "Oz", "Pz", "P3", "P4", "POz", "P1"]
    channel_mapping = dict(zip(eeg_channels, channel_names))

    # Show board information
    print(f"Sampling Rate: {sampling_rate}")
    print(f"Default Channels: {eeg_channels}")
    print(f"Channel Mapping: {channel_mapping}")

    board = BrainFlowBoardSetup(board_id, serial_port)
    board.setup()

    # Initialize stimulus
    stimulus = SSVEPStimulus(box_frequencies=frequencies, 
                             box_texts=buttons, 
                             box_text_indices=button_pos,          
                             display_index=display)

    # Start EEG data processing in a separate thread
    processing_thread = threading.Thread(target=run_processing, args=(board, stimulus, harmonics, sampling_rate, n_samples))
    processing_thread.start()

    # Run stimulus presentation in the main thread
    run_stimulus(stimulus)

    # Ensure the processing thread finishes before exiting
    processing_thread.join()

def run_stimulus(stimulus):
    stimulus.start()
    while not stimulus.is_finished():
        stimulus.update()
        time.sleep(1.0 / stimulus.refresh_rate)  # Maintain the refresh rate consistency

if __name__ == "__main__":
    main()
