import threading
from pynput import keyboard

class KeyListener:
    """
    A class to listen for keyboard events in a separate thread.

    Attributes:
        stop_flag (bool): A flag to indicate when to stop the listener.
    """

    def __init__(self):
        """
        Initializes the KeyListener with a stop flag set to False.
        """
        self.stop_flag = False

    def on_press(self, key):
        """
        Callback function that gets called when a key is pressed.

        Args:
            key (pynput.keyboard.Key): The key that was pressed.

        Returns:
            bool: False if the Esc key was pressed, which stops the listener.
        """
        if key == keyboard.Key.esc:
            self.stop_flag = True
            return False  # Stop listener

    def start_listener(self):
        """
        Starts the keyboard listener and waits for a key press event.
        """
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        listener.join()  # Ensure the listener thread waits for key press

    def run_listener(self):
        """
        Runs the keyboard listener in a separate thread.
        """
        listener_thread = threading.Thread(target=self.start_listener)
        listener_thread.start()
