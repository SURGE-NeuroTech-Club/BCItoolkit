import numpy as np
from psychopy import visual, event, core, monitors
import time

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

    def is_finished(self):
        return not self.is_running

# Example Usage
def main():
    # Example parameters
    frequencies = [9.25, 11.25, 13.25, 15.25]
    buttons = ['Right', 'Left', 'Up', 'Down']
    display = 0
    
    # Instantiate and start stimulus
    stimulus = SSVEPStimulus(box_frequencies=frequencies, box_texts=buttons, display_index=display)
    stimulus.start()

if __name__ == "__main__":
    main()
