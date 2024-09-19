import time
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color


sphero_list = scanner.find_toys()


# Check if the list contains any toys
if sphero_list:
    # Select the first toy from the list
    sphero = sphero_list[0]
else:
    print("No Sphero toys found.")




with SpheroEduAPI(sphero) as droid:
    droid.set_main_led(Color(r=0, g=255, b=0))    
    droid.set_speed(60)
    time.sleep(2)
    droid.set_speed(0)