from psychopy import visual, event, core, monitors
import numpy as np
from multiprocessing import Process
import time
import warnings
warnings.filterwarnings("ignore", message="elementwise comparison failed; returning scalar instead")

class SSVEPStimulus:
    """
    Class to create and run a Steady-State Visual Evoked Potential (SSVEP) stimulus using PsychoPy.
    
    Attributes:
        box_frequencies (list): List of desired frequencies for the stimulus boxes.
        box_texts (list): List of texts to display inside the boxes.
        box_text_indices (list): List of indices corresponding to the boxes that should display text.
        display_mode (str): Mode of display ('freq', 'text', 'both', or None).
        win (visual.Window): PsychoPy window object.
        refresh_rate (float): Measured refresh rate of the display.
        actual_frequencies (list): List of actual frequencies adjusted to the refresh rate.
        boxes (list): List of dictionaries containing box information.
        frame_count (int): Counter for the number of frames displayed.
        has_started (bool): Flag indicating whether the stimulus has started.
        start_button (visual.Rect): Start button rectangle.
        start_text (visual.TextStim): Start button text.
    """
    
    def __init__(self, box_frequencies, queue,  box_texts=None, box_text_indices=None, display_index=0, display_mode=None, monitor_name='testMonitor'):
        """
        Initializes the SSVEPStimulus class with the given parameters.
        
        Args:
            box_frequencies (list): List of desired frequencies for the stimulus boxes.
            box_texts (list, optional): List of texts to display inside the boxes. Defaults to None.
            box_text_indices (list, optional): List of indices corresponding to the boxes that should display text. Defaults to None.
            display_index (int, optional): Index of the display screen to use. Defaults to 0.
            display_mode (str, optional): Mode of display ('freq', 'text', 'both', or None). Defaults to None.
            monitor_name (str, optional): Name of the monitor configuration to use. Defaults to 'testMonitor'.
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

        refresh_rates = []
        for _ in range(2):
            refresh_rate = self.win.getActualFrameRate(nIdentical=80, nWarmUpFrames=200, threshold=1)
            if refresh_rate is not None:
                refresh_rates.append(refresh_rate)
            core.wait(0.1)

        self.refresh_rate = round(np.mean(refresh_rates), 0)  
        print(f"Measured Refresh Rate: {self.refresh_rate:.2f} Hz")

        self.actual_frequencies = self.calculate_actual_frequencies(box_frequencies)

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

        self.boxes = []
        self.frame_count = 0

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

            if self.display_mode in ["both", "text"] and self.box_texts and idx in box_text_indices:
                box_text = box_texts[box_text_indices.index(idx)]
                text_pos = (pos[0], pos[1] + 30) if self.display_mode == "both" else pos
                box_text_stim = visual.TextStim(win=self.win, text=box_text, color='black', pos=text_pos)
                box_info["box_text"] = box_text_stim
            
            self.boxes.append(box_info)

        self.has_started = False
        self.start_button = visual.Rect(win=self.win, width=300, height=100, fillColor='green', pos=(0, 0))
        self.start_text = visual.TextStim(win=self.win, text='Press Space/Enter to Start', color='white', pos=(0, 0))

    def calculate_actual_frequencies(self, desired_frequencies):
        """
        Calculates the actual frequencies based on the desired frequencies and the refresh rate.
        
        Args:
            desired_frequencies (list): List of desired frequencies.
        
        Returns:
            list: List of actual frequencies adjusted to the refresh rate.
        """
        global actual_frequencies
        actual_frequencies = []
        for freq in desired_frequencies:
            frames_per_cycle = round(self.refresh_rate / freq)
            actual_freq = round(self.refresh_rate / frames_per_cycle, 2)
            actual_frequencies.append(actual_freq)
        self.queue.put(actual_frequencies)
        return actual_frequencies

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

def start_ssvep_stimulus(box_frequencies, box_texts=None, box_text_indices=None, display_index=0, display_mode=None, monitor_name='testMonitor'):
    """
    Starts the SSVEP stimulus in the current process.
    
    Args:
        box_frequencies (list): List of desired frequencies for the stimulus boxes.
        box_texts (list, optional): List of texts to display inside the boxes. Defaults to None.
        box_text_indices (list, optional): List of indices corresponding to the boxes that should display text. Defaults to None.
        display_index (int, optional): Index of the display screen to use. Defaults to 0.
        display_mode (str, optional): Mode of display ('freq', 'text', 'both', or None). Defaults to None.
        monitor_name (str, optional): Name of the monitor configuration to use. Defaults to 'testMonitor'.
    """
    stimulus = SSVEPStimulus(box_frequencies, box_texts, box_text_indices, display_index, display_mode, monitor_name)
    stimulus.run()

def run_ssvep_stimulus_in_process(box_frequencies, box_texts=None, box_text_indices=None, display_index=0, display_mode=None, monitor_name='testMonitor'):
    """
    Starts the SSVEP stimulus in a separate process.
    
    Args:
        box_frequencies (list): List of desired frequencies for the stimulus boxes.
        box_texts (list, optional): List of texts to display inside the boxes. Defaults to None.
        box_text_indices (list, optional): List of indices corresponding to the boxes that should display text. Defaults to None.
        display_index (int, optional): Index of the display screen to use. Defaults to 0.
        display_mode (str, optional): Mode of display ('freq', 'text', 'both', or None). Defaults to None.
        monitor_name (str, optional): Name of the monitor configuration to use. Defaults to 'testMonitor'.
    
    Returns:
        Process: The process running the SSVEP stimulus.
    """
    stimulus_process = Process(target=start_ssvep_stimulus, args=(box_frequencies, box_texts, box_text_indices, display_index, display_mode, monitor_name))
    stimulus_process.start()
    return stimulus_process

# Automatically start the SSVEP stimulus in a separate process when the module is imported
if __name__ == "__main__":
    box_frequencies = [9.25, 11.25, 13.25, 15.25]
    box_texts = ['Right', 'Left', 'Up', 'Down']
    box_text_indices = [0, 1, 2, 3]
    display_index = 0
    display_mode = 'both'
    # monitor_name = 'testMonitor'   

    # Using `run_ssvep_stimulus_in_process`:
    run_ssvep_stimulus_in_process(box_frequencies, 
                                    box_texts=box_texts, 
                                    box_text_indices=box_text_indices,          
                                    display_mode=display_mode)
    
    time.sleep(20)
    print(actual_frequencies)

    # Using `start_ssvep_stimulus`: (blocking!)
    # start_ssvep_stimulus(box_frequencies, 
    #                     box_texts=box_texts, 
    #                     box_text_indices=box_text_indices,          
    #                     display_mode=display_mode)
    
    # Using default class method: (blocking!)
    # stimulus = SSVEPStimulus(box_frequencies, 
    #                          box_texts=box_texts, 
    #                          box_text_indices=box_text_indices,          
    #                          display_mode=display_mode)
    # stimulus.start()