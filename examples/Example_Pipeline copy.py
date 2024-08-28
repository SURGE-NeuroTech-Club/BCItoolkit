import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
from psychopy import visual, event, core, monitors
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from mvlearn.embed import CCA
from multiprocessing import Process, Queue

# Assuming the other modules (brainflow_stream, filtering, segmentation, classification) are available
from modules.brainflow_stream import BrainFlowBoardSetup
from modules.filtering import *
from modules.segmentation import Segmentation
from modules.classification import SSVEPClassifier

class SSVEPStimulus:
    def __init__(self, box_frequencies, box_texts, box_text_indices, display_mode, display_index):
        self.box_frequencies = box_frequencies
        self.box_texts = box_texts
        self.box_text_indices = box_text_indices
        self.display_mode = display_mode
        self.display_index = display_index

        # Setup PsychoPy window and monitor
        monitor = monitors.Monitor('testMonitor')
        self.win = visual.Window(
            monitor=monitor,
            screen=display_index,
            fullscr=False,
            size=(800, 600),
            color='black',
            units='pix',
            allowGUI=False,
            winType='pyglet',
            autoLog=False
        )

        self.refresh_rate = self.win.getActualFrameRate() or 60  # Use default if unable to measure
        print(f"Measured Refresh Rate: {self.refresh_rate:.2f} Hz")

        self.boxes = self.create_boxes()

        self.is_running = False
        self.frame_count = 0

    def create_boxes(self):
        boxes = []
        radius = min(self.win.size) // 3
        centerX, centerY = 0, 0
        num_boxes = len(self.box_frequencies)

        for i, freq in enumerate(self.box_frequencies):
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

def start_visual_stimulus(box_frequencies, box_texts, box_text_indices, display_mode, display_index):
    stimulus = SSVEPStimulus(box_frequencies, box_texts, box_text_indices, display_mode, display_index)
    stimulus.start()

def run_eeg_processing(serial_port, board_id, frequencies, buttons, button_pos, segment_duration):
    board = BrainFlowBoardSetup(board_id, serial_port)
    board.setup()

    segmentation_time_wait = Segmentation(board, segment_duration=2)
    actual_freqs = frequencies

    cca_classifier = SSVEPClassifier(frequencies=actual_freqs, 
                                     harmonics=np.arange(1, 4), 
                                     sampling_rate=BoardShim.get_sampling_rate(board_id), 
                                     n_samples=BoardShim.get_sampling_rate(board_id) * segment_duration, 
                                     method='CCA', 
                                     stack_harmonics=True)

    while True:
        segment = segmentation_time_wait.get_segment_time()
        if segment is None:
            time.sleep(0.1)  # Wait a little before trying again
            continue

        eeg_segment = segment[:8, :]  # Assuming 8 channels, adjust if necessary
        detected_freq, correlation = cca_classifier(eeg_segment)
        print(f"Detected frequency using CCA: {detected_freq} Hz with correlation: {correlation:.3f}")

        time.sleep(0.1)

def main():
    serial_port = 'COM4'
    board_id = BoardIds.SYNTHETIC_BOARD
    frequencies = [9.25, 11.25, 13.25, 15.25]
    buttons = ['Right', 'Left', 'Up', 'Down']
    button_pos = [0, 2, 3, 1]
    segment_duration = 4
    display = 0

    # Start the visual stimulus in a separate process
    stimulus_process = Process(target=start_visual_stimulus, args=(frequencies, buttons, button_pos, "both", display))
    stimulus_process.start()

    # Start the EEG processing in a separate process
    eeg_process = Process(target=run_eeg_processing, args=(serial_port, board_id, frequencies, buttons, button_pos, segment_duration))
    eeg_process.start()

    # Wait for both processes to complete
    stimulus_process.join()
    eeg_process.join()

if __name__ == "__main__":
    main()