from psychopy import visual, event, core, monitors
import numpy as np
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

    def __init__(self, box_frequencies, box_texts=None, box_text_indices=None, display_index=0, display_mode="both", monitor_name='testMonitor'):
        """
        Initializes the SSVEPStimulus class.

        Args:
            box_frequencies (list of float): Desired frequencies for the flickering boxes.
            box_texts (list of str, optional): Texts or symbols to display on the boxes.
            box_text_indices (list of int, optional): Indices indicating which boxes should display text.
            display_index (int, optional): Index of the display screen to use.
            display_mode (str, optional): Specifies what to display on the boxes. 
                                          Options are "freq" for frequency, "text" for text, "both" for both frequency and text, or "None" for empty boxes.
            monitor_name (str, optional): Name of the monitor configuration to use in PsychoPy.
        """
        self.box_frequencies = box_frequencies
        self.box_texts = box_texts or []
        self.box_text_indices = box_text_indices or []
        self.display_mode = display_mode

        if self.box_texts and len(self.box_texts) != len(self.box_text_indices):
            raise ValueError("The length of box_texts and box_text_indices must be the same if box_texts is provided.")

        # Setup monitor and window
        self.win = visual.Window(
            monitor=monitors.Monitor(name=monitor_name),
            screen=display_index,
            fullscr=True,
            color='black',
            units='pix',
            allowGUI=False,
            winType='pyglet',
            autoLog=False
        )

        # Measure screen refresh rate
        self.refresh_rate = round(self.win.getActualFrameRate(nIdentical=80, nWarmUpFrames=120, threshold=1), 0)
        print(f"Measured Refresh Rate: {self.refresh_rate:.2f} Hz")

        # Calculate and assign frequencies to boxes
        self.actual_frequencies = self.calculate_actual_frequencies(box_frequencies)
        print(f"Actual Frequencies: {self.actual_frequencies}")
        self.boxes = self.create_boxes()

        # Start presentation flag and UI elements
        self.start = False
        self.start_button = visual.Rect(win=self.win, width=300, height=100, fillColor='green', pos=(0, 0))
        self.start_text = visual.TextStim(win=self.win, text='Press Space/Enter to Start', color='white', pos=(0, 0))
        self.frame_count = 0

    def calculate_actual_frequencies(self, desired_frequencies):
        """Calculate the actual frequencies based on the refresh rate."""
        return [round(self.refresh_rate / round(self.refresh_rate / freq), 2) for freq in desired_frequencies]

    def create_boxes(self):
        """Create visual boxes and assign properties based on the calculated frequencies and display mode."""
        centerX, centerY = 0, 0
        radius = min(self.win.size) // 3
        num_boxes = len(self.actual_frequencies)

        sorted_indices = sorted(range(num_boxes), key=lambda i: self.actual_frequencies[i])
        interleaved_indices = [sorted_indices[i] if i % 2 == 0 else sorted_indices[-(i//2+1)] for i in range(num_boxes)]

        boxes = []
        for i, idx in enumerate(interleaved_indices):
            angle = 2 * np.pi * i / num_boxes
            pos = (centerX + int(radius * np.cos(angle)), centerY + int(radius * np.sin(angle)))
            box = visual.Rect(win=self.win, width=150, height=150, fillColor='white', lineColor='white', pos=pos)

            box_info = {"box": box, "frequency": self.actual_frequencies[idx], "frame_count": 0, "on": True}
            if self.display_mode in ["freq", "both"]:
                box_info["text"] = visual.TextStim(win=self.win, text=f"{self.actual_frequencies[idx]:.2f} Hz", color='black', pos=pos)
            if self.display_mode in ["text", "both"] and idx in self.box_text_indices:
                offset = 30 if self.display_mode == "both" else 0
                box_info["box_text"] = visual.TextStim(win=self.win, text=self.box_texts[self.box_text_indices.index(idx)], color='black', pos=(pos[0], pos[1] + offset))

            boxes.append(box_info)
        return boxes

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

            if not self.start:
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

# Example usage
if __name__ == "__main__":
    box_frequencies = [9, 9.2, 9.5, 10, 10.3, 11]
    box_texts = ["A", "B", "C"]
    box_text_indices = [0, 2, 4]
    display_mode = "text"

    stimulus = SSVEPStimulus(box_frequencies, box_texts=box_texts, box_text_indices=box_text_indices, display_mode=display_mode)
    print(f"Requested Frequencies: {stimulus.box_frequencies}")
    # print(f"Calculated Frequencies: {stimulus.get_actual_frequencies()}")

    stimulus.run()
