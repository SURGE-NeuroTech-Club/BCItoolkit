from psychopy import visual, core, monitors
import numpy as np

def get_possible_frequencies(display_index=0, num_measurements=5, monitor_name='testMonitor'):
    """
    Measures the refresh rate of the monitor multiple times, averages the results,
    and returns a list of possible frequencies it can display, rounded to 2 decimal places.

    Args:
        display_index (int, optional): The index of the display screen to use.
        num_measurements (int, optional): The number of times to measure the refresh rate to reduce variability.
        monitor_name (str, optional): The name of the monitor configuration to use in PsychoPy.

    Returns:
        dict: A dictionary with the average refresh rate and a list of possible frequencies that can be displayed,
              each rounded to 2 decimal places.
    """
    # Setup monitor and window
    monitor = monitors.Monitor(name=monitor_name)  # Change as appropriate
    
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

    # Display instructions to the user before measurement
    instruction_text = visual.TextStim(win, text="The script will now measure the refresh rate.\nYour screen will be black for several seconds.", color='white', pos=(0, 0), height=50)
    instruction_text.draw()
    win.flip()

    # Pause to give the user time to read the instructions
    core.wait(3)

    # Measure screen refresh rate multiple times and average
    refresh_rates = []
    for _ in range(num_measurements):
        refresh_rate = win.getActualFrameRate(nIdentical=80, nWarmUpFrames=120, threshold=1)
        refresh_rates.append(refresh_rate)
        core.wait(0.1)  # Short delay between measurements to ensure variability is captured

    avg_refresh_rate = np.mean(refresh_rates)
    print(f"Average Measured Refresh Rate: {avg_refresh_rate:.2f} Hz")

    # Calculate all possible frequencies, rounded to 2 decimal places
    possible_frequencies = []
    for frames_per_cycle in range(1, int(avg_refresh_rate) + 1):
        freq = avg_refresh_rate / frames_per_cycle
        possible_frequencies.append(round(freq, 2))

    win.close()

    return {
        "average_refresh_rate": round(avg_refresh_rate, 2),
        "possible_frequencies": possible_frequencies
    }

# Example usage
if __name__ == "__main__":
    frequencies_info = get_possible_frequencies()
    print(f"Average Refresh Rate: {frequencies_info['average_refresh_rate']:.2f} Hz")
    print("Possible Frequencies:")
    for freq in frequencies_info['possible_frequencies']:
        print(f"{freq:.2f} Hz")
