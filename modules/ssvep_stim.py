from psychopy import visual, event, core, monitors
import numpy as np
from multiprocessing import Process, Queue
import time
import warnings

warnings.filterwarnings("ignore", message="elementwise comparison failed; returning scalar instead")

class SSVEPStimulus:
    """
    Class to create and run a Steady-State Visual Evoked Potential (SSVEP) stimulus using PsychoPy.
    """
    
    def __init__(self, box_frequencies, queue=None, box_texts=None, box_text_indices=None, display_index=0, display_mode="freq", monitor_name="testMonitor"):
        """
        Initializes the SSVEPStimulus class with the given parameters.
        
        Parameters:
        - box_frequencies: List of frequencies for the boxes.
        - queue: Optional multiprocessing queue to communicate actual frequencies.
        - box_texts: Optional list of texts to display on the boxes.
        - box_text_indices: Optional list of indices corresponding to the box_texts.
        - display_index: Index of the display screen to use.
        - display_mode: Mode of display ('freq', 'text', 'both').
        - monitor_name: Name of the monitor configuration to use.
        """
        self.box_frequencies = box_frequencies
        self.box_texts = box_texts
        self.box_text_indices = box_text_indices
        self.display_mode = display_mode
        self.queue = queue

        if box_texts and len(box_texts) != len(box_text_indices):
            raise ValueError("The length of box_texts and box_text_indices must be the same if box_texts is provided.")
        
        monitor = monitors.Monitor(name=monitor_name)
        self.win = visual.Window(
            monitor=monitor, 
            screen=display_index, 
            fullscr=True, 
            color='black', 
            units='pix', 
            allowGUI=False, 
            winType='pyglet',
            autoLog=False
        )

        self.refresh_rate = self._measure_refresh_rate()
        print(f"Measured Refresh Rate: {self.refresh_rate:.2f} Hz")

        self.actual_frequencies = self.calculate_actual_frequencies(box_frequencies)
        self.boxes = self._create_boxes()
        self.frame_count = 0
        self.has_started = False
        self.start_button = visual.Rect(win=self.win, width=300, height=100, fillColor='green', pos=(0, 0))
        self.start_text = visual.TextStim(win=self.win, text='Press Space/Enter to Start', color='white', pos=(0, 0))

    def _measure_refresh_rate(self):
        """
        Measures the refresh rate of the display.
        
        Returns:
        - The measured refresh rate.
        """
        refresh_rates = []
        for _ in range(2):
            refresh_rate = self.win.getActualFrameRate(nIdentical=80, nWarmUpFrames=200, threshold=1)
            if refresh_rate is not None:
                refresh_rates.append(refresh_rate)
            core.wait(0.1)

        if refresh_rates:
            return round(np.mean(refresh_rates), 0)
        else:
            print("Warning: Could not measure a consistent refresh rate. Using default value of 60 Hz.")
            return 60  # Default refresh rate

    def calculate_actual_frequencies(self, desired_frequencies):
        """
        Calculates the actual frequencies based on the desired frequencies and the refresh rate.
        
        Parameters:
        - desired_frequencies: List of desired frequencies.
        
        Returns:
        - List of actual frequencies.
        """
        actual_frequencies = []
        for freq in desired_frequencies:
            frames_per_cycle = round(self.refresh_rate / freq)
            actual_freq = round(self.refresh_rate / frames_per_cycle, 2)
            actual_frequencies.append(actual_freq)
        if self.queue:
            self.queue.put(actual_frequencies)
        return actual_frequencies

    def _create_boxes(self):
        """
        Creates the visual boxes for the stimulus.
        
        Returns:
        - List of box information dictionaries.
        """
        sorted_indices = sorted(range(len(self.actual_frequencies)), key=lambda i: self.actual_frequencies[i])
        interleaved_indices = []
        left, right = 0, len(sorted_indices) - 1
        while left <= right:
            if left == right:
                interleaved_indices.append(sorted_indices[left])
            else:
                interleaved_indices.append(sorted_indices[left])
                interleaved_indices.append(sorted_indices[right])
            left += 1
            right -= 1

        boxes = []
        centerX, centerY = 0, 0
        radius = min(self.win.size) // 3
        num_boxes = len(self.actual_frequencies)

        for i, idx in enumerate(interleaved_indices):
            angle = 2 * np.pi * i / num_boxes
            pos = (centerX + int(radius * np.cos(angle)), centerY + int(radius * np.sin(angle)))
            box = visual.Rect(win=self.win, width=150, height=150, fillColor='white', lineColor='white', pos=pos)
            
            box_info = {
                "box": box,
                "frequency": self.actual_frequencies[idx],
                "frame_count": 0,
                "on": True
            }

            if self.display_mode in ["freq", "both"]:
                freq_text_stim = visual.TextStim(win=self.win, text=f"{self.actual_frequencies[idx]:.2f} Hz", color='black', pos=pos)
                box_info["text"] = freq_text_stim

            if self.display_mode in ["both", "text"] and self.box_texts and idx in self.box_text_indices:
                box_text = self.box_texts[self.box_text_indices.index(idx)]
                text_pos = (pos[0], pos[1] + 30) if self.display_mode == "both" else pos
                box_text_stim = visual.TextStim(win=self.win, text=box_text, color='black', pos=text_pos)
                box_info["box_text"] = box_text_stim
            
            boxes.append(box_info)
        
        return boxes

    def run(self):
        """
        Runs the SSVEP stimulus, handling the display and flickering of the boxes.
        """
        while True:
            keys = event.getKeys()
            if 'escape' in keys:
                break
            elif 'space' in keys or 'return' in keys:
                self.has_started = True
            
            if not self.has_started:
                self.start_button.draw()
                self.start_text.draw()
            else:
                self.frame_count += 1
                for box in self.boxes:
                    flicker_period = self.refresh_rate / box["frequency"]
                    if (self.frame_count % flicker_period) < (flicker_period / 2):
                        if not box["on"]:
                            box["on"] = True
                            box["box"].setAutoDraw(True)
                            if "text" in box:
                                box["text"].setAutoDraw(True)
                            if "box_text" in box:
                                box["box_text"].setAutoDraw(True)
                    else:
                        if box["on"]:
                            box["on"] = False
                            box["box"].setAutoDraw(False)
                            if "text" in box:
                                box["text"].setAutoDraw(False)
                            if "box_text" in box:
                                box["box_text"].setAutoDraw(False)

            self.win.flip()
        
        self.win.close()
        core.quit()

    def stop(self):
        """
        Stops the SSVEP stimulus and closes the PsychoPy window.
        """
        self.has_started = False
        self.win.close()
        core.quit()

def start_ssvep_stimulus(box_frequencies, queue=None, box_texts=None, box_text_indices=None, display_index=0, display_mode=None, monitor_name='testMonitor'):
    """
    Starts the SSVEP stimulus in the current process.
    
    Parameters:
    - box_frequencies: List of frequencies for the boxes.
    - queue: Optional multiprocessing queue to communicate actual frequencies.
    - box_texts: Optional list of texts to display on the boxes.
    - box_text_indices: Optional list of indices corresponding to the box_texts.
    - display_index: Index of the display screen to use.
    - display_mode: Mode of display ('freq', 'text', 'both').
    - monitor_name: Name of the monitor configuration to use.
    """
    stimulus = SSVEPStimulus(box_frequencies, queue, box_texts, box_text_indices, display_index, display_mode, monitor_name)
    stimulus.run()

class SSVEPStimulusRunner:
    """
    Class to manage the SSVEP stimulus in a separate process.
    """
    
    def __init__(self, box_frequencies, box_texts=None, box_text_indices=None, display_index=0, display_mode=None, monitor_name='testMonitor'):
        """
        Initializes the SSVEPStimulusRunner class with the given parameters.
        
        Parameters:
        - box_frequencies: List of frequencies for the boxes.
        - box_texts: Optional list of texts to display on the boxes.
        - box_text_indices: Optional list of indices corresponding to the box_texts.
        - display_index: Index of the display screen to use.
        - display_mode: Mode of display ('freq', 'text', 'both').
        - monitor_name: Name of the monitor configuration to use.
        """
        self.box_frequencies = box_frequencies
        self.box_texts = box_texts
        self.box_text_indices = box_text_indices
        self.display_index = display_index
        self.display_mode = display_mode
        self.monitor_name = monitor_name
        self.queue = Queue()
        self.process = None

    def start(self):
        """
        Starts the SSVEP stimulus in a separate process.
        """
        self.process = Process(target=start_ssvep_stimulus, args=(self.box_frequencies, self.queue, self.box_texts, self.box_text_indices, self.display_index, self.display_mode, self.monitor_name))
        self.process.start()

    def get_actual_frequencies(self):
        """
        Retrieves the actual frequencies from the queue.
        
        Returns:
        - List of actual frequencies.
        """
        if self.process and self.process.is_alive():
            time.sleep(2)  # Wait for some time or until the frequencies are available
            actual_freqs = self.queue.get()  # Retrieve the frequencies from the queue
            return actual_freqs
        else: 
            raise RuntimeError("The process isn't alive - cannot return actual_frequencies. Use: start() or is_running()")

    def stop(self):
        """
        Stops the SSVEP stimulus process.
        """
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def is_running(self):
        """
        Checks if the SSVEP stimulus process is still running.
        
        Returns:
        - True if the process is running, False otherwise.
        """
        return self.process.is_alive() if self.process else False


# Example usage
if __name__ == "__main__":
    box_frequencies = [9.25, 11.25, 13.25, 15.25]
    box_texts = ['Right', 'Left', 'Up', 'Down']
    box_text_indices = [0, 1, 3, 2]
    display_index = 0
    display_mode = 'both'
    monitor_name = 'testMonitor'
    
    
    # Using default method: (blocking!) -> use this if running the SSVEPStimulus from a seperate script!
    # start_ssvep_stimulus(box_frequencies=box_frequencies, 
    #                      box_texts=box_texts, 
    #                      box_text_indices=box_text_indices,
    #                      display_index=display_index, 
    #                      display_mode=display_mode, 
    #                      monitor_name = monitor_name)
    
    
    # Start the SSVEP stimulus in a separate process -> use this if running the SSVEPStimulus in the same script as EEG processing or other code
    # Create an instance of the SSVEPStimulusRunner class
    stimulus_process = SSVEPStimulusRunner(box_frequencies = box_frequencies, 
                                            box_texts = box_texts, 
                                            box_text_indices = box_text_indices,
                                            display_index = display_index, 
                                            display_mode = display_mode, 
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