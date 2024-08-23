from psychopy import visual, event, core, monitors
import numpy as np

# Bug that spits futurewarning but doesn't seem to affect execution
import warnings
warnings.filterwarnings("ignore", message="elementwise comparison failed; returning scalar instead")

class SSVEPStimulus:
    """
    Class to handle the stimulus presentation paradigm for an SSVEP BCI system using flickering boxes.
    
    Attributes:
        win (visual.Window): The PsychoPy window where stimuli are presented.
        refresh_rate (float): The refresh rate of the display in Hz.
        actual_frequencies (list of float): The list of actual frequencies that can be displayed given the refresh rate.
        boxes (list of dict): A list of dictionaries, each containing the box stimulus and its related properties.
        start (bool): Flag indicating whether the stimulus presentation has started.
        frame_count (int): Counter for the number of frames elapsed.
    """

    def __init__(self, box_frequencies, box_texts=None, box_text_indices=None, display_index=0, display_mode=None):
        """
        Initializes the SSVEPStimulus class.

        Args:
            box_frequencies (list of float): Desired frequencies for the flickering boxes.
            box_texts (list of str, optional): Texts or symbols to display on the boxes.
            box_text_indices (list of int, optional): Indices indicating which boxes should display text.
            display_index (int, optional): Index of the display screen to use.
            display_mode (str, optional): Specifies what to display on the boxes. 
                                          Options are "freq" for frequency, "text" for text, "both" for both frequency and text, or "None" for empty boxes.
        """
        if box_texts and len(box_texts) != len(box_text_indices):
            raise ValueError("The length of box_texts and box_text_indices must be the same if box_texts is provided.")
        
        # Setup monitor and window
        monitor = monitors.Monitor(name='testMonitor')  # Change as appropriate
        
        win = visual.Window(
            monitor=monitor, 
            screen=display_index, 
            fullscr=True, 
            color='black', 
            units='pix', 
            allowGUI=False, 
            winType='pyglet',
            autoLog=False
        )

        self.win = win

        # Measure screen refresh rate using getActualFrameRate      
        self.refresh_rate = win.getActualFrameRate(nIdentical=80, nWarmUpFrames=120, threshold=1)
        print(f"Measured Refresh Rate: {self.refresh_rate:.2f} Hz")

        # Calculate actual frequencies that can be shown given the refresh rate
        self.actual_frequencies = self.calculate_actual_frequencies(box_frequencies)
        print(f"Actual Frequencies: {self.actual_frequencies}")

        # Sort and interleave the indices of the boxes to maximize the distance between similar frequencies
        sorted_indices = sorted(range(len(self.actual_frequencies)), key=lambda i: self.actual_frequencies[i])
        interleaved_indices = []

        left = 0
        right = len(sorted_indices) - 1
        while left <= right:
            if left == right:
                interleaved_indices.append(sorted_indices[left])
            else:
                interleaved_indices.append(sorted_indices[left])
                interleaved_indices.append(sorted_indices[right])
            left += 1
            right -= 1

        # Create boxes and assign frequencies
        self.boxes = []
        self.box_texts = box_texts
        self.display_mode = display_mode
        self.frame_count = 0
        
        centerX, centerY = 0, 0  # Center of the screen
        radius = min(win.size) // 3  # Radius for circular placement
        num_boxes = len(self.actual_frequencies)

        for i, idx in enumerate(interleaved_indices):
            angle = 2 * np.pi * i / num_boxes
            pos = (centerX + int(radius * np.cos(angle)), centerY + int(radius * np.sin(angle)))

            box = visual.Rect(win=win, width=150, height=150, fillColor='white', lineColor='white', pos=pos)
            
            box_info = {
                "box": box,
                "frequency": self.actual_frequencies[idx],  # Always include the frequency
                "frame_count": 0,
                "on": True  # Track whether the box is on or off
            }

            # Add frequency text if the mode is 'freq' or 'both'
            if self.display_mode in ["freq", "both"]:
                freq_text_stim = visual.TextStim(win=win, text=f"{self.actual_frequencies[idx]:.2f} Hz", color='black', pos=pos)
                box_info["text"] = freq_text_stim

            # Add box text if specified for mode 'both'
            if self.display_mode in ["both"] and self.box_texts and idx in box_text_indices:
                box_text = box_texts[box_text_indices.index(idx)]
                box_text_stim = visual.TextStim(win=win, text=box_text, color='black', pos=(pos[0], pos[1] + 30))
                box_info["box_text"] = box_text_stim
                
            # Add centered box text if specified for mode 'text'
            if self.display_mode in ["text"] and self.box_texts and idx in box_text_indices:
                box_text = box_texts[box_text_indices.index(idx)]
                box_text_stim = visual.TextStim(win=win, text=box_text, color='black', pos=pos)
                box_info["box_text"] = box_text_stim
            
            self.boxes.append(box_info)
        
        self.start = False
        self.start_button = visual.Rect(win=win, width=300, height=100, fillColor='green', pos=(0, 0))
        self.start_text = visual.TextStim(win=win, text='Press Space/Enter to Start', color='white', pos=(0, 0))

    def calculate_actual_frequencies(self, desired_frequencies):
        """
        Calculate the actual frequencies that can be shown given the measured refresh rate.

        Args:
            desired_frequencies (list of float): The desired frequencies for the flickering boxes.

        Returns:
            list of float: The actual frequencies that can be achieved given the screen's refresh rate.
        """
        actual_frequencies = []
        for freq in desired_frequencies:
            frames_per_cycle = round(self.refresh_rate / freq)
            actual_freq = self.refresh_rate / frames_per_cycle
            actual_frequencies.append(actual_freq)
        return actual_frequencies

    def get_actual_frequencies(self):
        """
        Return the actual frequencies calculated for the stimulus presentation.
        
        Returns:
            list of float: The actual frequencies used.
        """
        return self.actual_frequencies

    def run(self):
        """
        Runs the main loop to handle the stimulus presentation.
        
        The loop waits for a key press (space/enter) to start the stimulus presentation. Once started,
        the boxes will flicker at their assigned frequencies. The loop will continue until the escape key is pressed.
        """
        while True:
            keys = event.getKeys()
            if 'escape' in keys:
                break
            elif 'space' in keys or 'return' in keys:
                self.start = True
            
            mouse = event.Mouse(visible=False, win=self.win)
            if not self.start:
                self.start_button.draw()
                self.start_text.draw()
            else:
                self.frame_count += 1
                for box in self.boxes:
                    flicker_period = self.refresh_rate / box["frequency"]
                    # Flip the state at the right time using frame counting
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

# Example usage
if __name__ == "__main__":
    # List of desired frequencies
    box_frequencies = [8, 10, 12, 14, 16, 18] 

    # List of texts or symbols to display on the stimuli
    box_texts = ["A", "B", "C"]  

    # Indices where the texts should be displayed
    box_text_indices = [0, 2, 4]  

    # Option to display just "freq", just "text", "both", or None (default)
    display_mode = "both"  

    stimulus = SSVEPStimulus(box_frequencies, 
                             box_texts=box_texts, 
                             box_text_indices=box_text_indices,          
                             display_mode=display_mode)
    
    # Retrieve and print the actual frequencies calculated (usefull for CCA analysis)
    actual_frequencies = stimulus.get_actual_frequencies()
    print(f"Calculated Frequencies: {actual_frequencies}")

    # Run the stimulus presentation
    stimulus.run()
