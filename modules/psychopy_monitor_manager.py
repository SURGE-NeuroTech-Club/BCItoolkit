from psychopy import monitors
from screeninfo import get_monitors

class MonitorManager:
    """
    Class to manage monitor creation and configuration in PsychoPy.
    Automatically detects resolution and width.
    """
    
    def __init__(self, monitor_name, create_monitor=False, monitor_params=None, monitor_index=0):
        """
        Initializes the MonitorManager class.
        
        Parameters:
        - monitor_name: Name of the monitor configuration to use.
        - create_monitor: Whether to create the monitor if it doesn't exist.
        - monitor_params: Dictionary of additional monitor parameters like width, distance.
        - monitor_index: Index of the monitor to auto-detect (in case multiple monitors are attached).
        """
        self.monitor_name = monitor_name
        self.create_monitor = create_monitor
        self.monitor_params = monitor_params if monitor_params else {}
        self.monitor_index = monitor_index  # Index to select a monitor from the detected list

    def auto_detect_monitor_settings(self):
        """
        Automatically detects resolution and width using screeninfo.
        """
        monitors_info = get_monitors()
        
        # Check if the specified monitor index is valid
        if self.monitor_index >= len(monitors_info):
            raise ValueError(f"Invalid monitor index: {self.monitor_index}. Only {len(monitors_info)} monitor(s) detected.")
        
        selected_monitor = monitors_info[self.monitor_index]

        resolution = [selected_monitor.width, selected_monitor.height]
        width = selected_monitor.width_mm / 10.0  # Convert from mm to cm

        return {
            'resolution': resolution,
            'width': width,
        }

    def get_or_create_monitor(self):
        """
        Checks if the monitor exists, and if not, creates it with auto-detected or specified parameters.
        
        Returns:
        - The PsychoPy monitor object.
        """
        monitor_list = monitors.getAllMonitors()

        if self.monitor_name not in monitor_list:
            if self.create_monitor:
                print(f"Monitor '{self.monitor_name}' not found. Creating a new monitor with auto-detected parameters.")
                
                auto_settings = self.auto_detect_monitor_settings()

                new_monitor = monitors.Monitor(self.monitor_name)
                new_monitor.setWidth(self.monitor_params.get('width', auto_settings['width']))  # Default to auto-detected width
                new_monitor.setDistance(self.monitor_params.get('distance', 60.0))  # Default distance to 60 cm
                new_monitor.setSizePix(self.monitor_params.get('resolution', auto_settings['resolution']))  # Auto-detected resolution
                new_monitor.saveMon()  # Save the monitor
            else:
                raise ValueError(f"Monitor '{self.monitor_name}' does not exist, and create_monitor is set to False.")
        else:
            print(f"Monitor '{self.monitor_name}' found.")
            new_monitor = monitors.Monitor(self.monitor_name)
        
        return new_monitor


if __name__ == "__main__":
    monitor_name = 'AutoMonitor2'
    create_monitor = True
    monitor_index = 1  # Change this to select the second monitor (index 0 for the second monitor)
    monitor_params = {
        'distance': 60.0  # You can specify distance, others will be auto-detected
    }
    
    # Create an instance of MonitorManager with the monitor_index parameter
    monitor_manager = MonitorManager(monitor_name, create_monitor, monitor_params, monitor_index)
    
    # Get or create the monitor
    monitor = monitor_manager.get_or_create_monitor()

    # Now you can use the monitor in your PsychoPy experiment
    print(f"Monitor '{monitor_name}' is ready with the following parameters:")
    print(f"Width: {monitor.getWidth()} cm")
    print(f"Distance: {monitor.getDistance()} cm")
    print(f"Resolution: {monitor.getSizePix()}")
