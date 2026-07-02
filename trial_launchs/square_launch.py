import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

# 1. SET YOUR URI
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E2')

def get_current_position(scf):
    """Captures the current LPS coordinates before takeoff"""
    log_config = LogConfig(name='Position', period_in_ms=10)
    log_config.add_variable('stateEstimate.x', 'float')
    log_config.add_variable('stateEstimate.y', 'float')
    log_config.add_variable('stateEstimate.z', 'float')

    pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            pos['x'] = data['stateEstimate.x']
            pos['y'] = data['stateEstimate.y']
            pos['z'] = data['stateEstimate.z']
            break 
    return pos

def run_mission():
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        
        print("Initializing... Waiting for LPS lock.")
        time.sleep(3) 

        # Identify Spawn Location
        spawn_pos = get_current_position(scf)
        print("-" * 40)
        print(f"SPAWN DETECTED AT: X={spawn_pos['x']:.2f}, Y={spawn_pos['y']:.2f}")
        print("-" * 40)

        # High-Level Commander handles takeoff and trajectory
        # default_height=0.5 ensures it takes off to 0.5m immediately
        with PositionHlCommander(scf, default_height=0.5) as pc:
            
            # STEP 1: Go to the Square Start Point
            print("Navigating from spawn to Square Start (0.0, 1.0, 0.5)...")
            pc.go_to(0.0, 1.0, 0.5)
            time.sleep(2)

            # STEP 2: Complete the Square
            print("Moving to Corner 2: (2.0, 1.0, 0.5)")
            pc.go_to(2.0, 1.0, 0.5)
            time.sleep(2)

            print("Moving to Corner 3: (2.0, 3.0, 0.5)")
            pc.go_to(2.0, 3.0, 0.5)
            time.sleep(2)

            print("Moving to Corner 4: (0.0, 3.0, 0.5)")
            pc.go_to(0.0, 3.0, 0.5)
            time.sleep(2)

            print("Closing the square: (0.0, 1.0, 0.5)")
            pc.go_to(0.0, 1.0, 0.5)
            time.sleep(2)

            # STEP 3: Landing
            print("Mission complete. Landing at start of square.")
            pc.land()

if _name_ == '_main_':
    run_mission()
