import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.positioning.position_hl_commander import PositionHlCommander

# 1. SET YOUR URI
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

def run_mission():
    cflib.crtp.init_drivers()

    # SyncCrazyflie opens the link and maintains the connection
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        
        # We DO NOT reset the estimator here. 
        # The LPS system will provide the absolute position automatically.
        # We just wait a moment for the drone to find its 'fix' from the anchors.
        print("Waiting for LPS position lock...")
        time.sleep(2) 

        # The High-Level Commander handles the takeoff and movement logic
        with PositionHlCommander(scf, default_height=0.5) as pc:
            
            # STEP 1: Fly from 'random' start point to the room origin
            print("Heading to Room Origin (0, 0, 0.5)...")
            pc.go_to(0.0, 0.0, 0.5)
            time.sleep(3) # Hover at (0,0) to show it found the spot
            # STEP 2: Execute your specific path

            print("Heading to Room Origin (0, 0, 0.5)...")
            pc.go_to(0.0, 0.0, 0.5)
            time.sleep(3)

            print("Moving to (0, 2, 0.5)...")
            pc.go_to(0.0, 2.0, 0.5)
            time.sleep(2)

            print("Heading to Room Origin (0, 0, 0.5)...")
            pc.go_to(0.0, 0.0, 0.5)
            time.sleep(3)

            print("Moving to (0, 4, 0.5)...")
            pc.go_to(0.0, 4.0, 0.5)
            time.sleep(2)

            # STEP 3: Return and land
            print("Returning to Origin for landing...")
            pc.go_to(0.0, 0.0, 0.5)
            pc.land()

if __name__ == '__main__':
    run_mission()
