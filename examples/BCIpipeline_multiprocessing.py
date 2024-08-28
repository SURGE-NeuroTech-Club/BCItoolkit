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

class VisualStimulus:
    def __init__(self, frequencies, display_index=0, monitor_name='testMonitor', size=(800, 600), fullscreen=False):
        self.frequencies = frequencies

        # Setup PsychoPy window and monitor
        monitor = monitors.Monitor(monitor_name)
        self.win = visual.Window(
            monitor=monitor,
            screen=display_index,
            fullscr=fullscreen,
            size=size,
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
        num_boxes = len(self.frequencies)

        for i, freq in enumerate(self.frequencies):
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

class BCIPipeline:
    def __init__(self, frequencies, harmonics, sampling_rate, n_samples):
        self.frequencies = frequencies
        self.harmonics = harmonics
        self.sampling_rate = sampling_rate
        self.n_samples = n_samples
        self.results_queue = Queue()

    def start(self):
        # Start EEG processing in a separate process
        processing_process = Process(target=self.run_processing, args=(self.results_queue,))
        processing_process.start()

        # Start monitoring results
        self.monitor_results()

        # Ensure the processing process finishes before exiting
        processing_process.join()

    def run_processing(self, results_queue):
        board = self.initialize_board()
        segmentation = Segmentation(board, segment_duration=2)
        classifier = SSVEPClassifier(frequencies=self.frequencies, 
                                     harmonics=self.harmonics, 
                                     sampling_rate=self.sampling_rate, 
                                     n_samples=self.n_samples, 
                                     method='CCA', 
                                     stack_harmonics=True)

        while True:
            segment = segmentation.get_segment_time()
            if segment is None:
                time.sleep(0.1)  # Wait a little before trying again
                continue

            eeg_segment = segment[:8, :]  # Assuming 8 channels, adjust if necessary
            detected_freq, correlation = classifier(eeg_segment)
            results_queue.put((detected_freq, correlation))

    def monitor_results(self):
        while True:
            if not self.results_queue.empty():
                result = self.results_queue.get()
                detected_freq, correlation = result
                print(f"Detected frequency: {detected_freq} Hz, Correlation: {correlation:.3f}")

    def initialize_board(self):
        serial_port = 'COM4'
        board_id = BoardIds.SYNTHETIC_BOARD
        board = BrainFlowBoardSetup(board_id, serial_port)
        board.setup()
        return board

def start_visual_stimulus(frequencies):
    stimulus = VisualStimulus(frequencies=frequencies, display_index=0, fullscreen=False)
    stimulus.start()

def main():
    # Setup parameters
    frequencies = [9.25, 11.25, 13.25, 15.25]
    harmonics = np.arange(1, 4)
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD)
    n_samples = sampling_rate * 4  # Example segment duration

    # Start the visual stimulus in a separate process
    stimulus_process = Process(target=start_visual_stimulus, args=(frequencies,))
    stimulus_process.start()

    # Initialize and run BCI pipeline
    pipeline = BCIPipeline(frequencies, harmonics, sampling_rate, n_samples)
    pipeline.start()

    # Wait for the stimulus to complete
    stimulus_process.join()

if __name__ == "__main__":
    main()