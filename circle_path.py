import time
import math
import numpy as np

import cflib.crtp

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper


# ==========================================
# RADIO URI
# ==========================================

URI = uri_helper.uri_from_env(
    default='radio://0/80/2M/E7E7E7E7E7'
)


# ==========================================
# TARGET CENTER
# ==========================================

X0 = 2.0
Y0 = 3.0
Z0 = 0.8

# Circle radius
R = 0.4


# ==========================================
# CIRCULAR TRAJECTORY
# Circle in Y-Z plane
# Revolution around X-axis
# ==========================================

def trajectory(t):

    x = X0

    y = Y0 + R * math.cos(t)

    z = Z0 + R * math.sin(t)

    return x, y, z


# ==========================================
# GENERATE TRAJECTORY POINTS
# ==========================================

def generate_trajectory_points():

    points = []

    # More points = smoother motion
    for t in np.linspace(0, 2 * math.pi, 160):

        points.append(trajectory(t))

    return points


# ==========================================
# FOLLOW TRAJECTORY
# ==========================================

def follow_trajectory(pc, points):

    for i, (x, y, z) in enumerate(points):

        print(f"\nPoint {i+1}")
        print(f"Target -> X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

        pc.go_to(x, y, z)

        # Slow and safe
        time.sleep(0.15)


# ==========================================
# MAIN MISSION
# ==========================================

def run_mission():

    print("Initializing Crazyflie drivers...")
    cflib.crtp.init_drivers()

    with SyncCrazyflie(
        URI,
        cf=Crazyflie(rw_cache='./cache')
    ) as scf:

        print("Connected")
        print("Waiting for stabilization...")

        time.sleep(3)

        # RANDOM SPAWN SAFE TAKEOFF
        # Drone can start anywhere
        with PositionHlCommander(
            scf,
            default_height=0.4,
            default_velocity=0.08,      # VERY SLOW
            default_landing_height=0.1
        ) as pc:

            print("Takeoff complete")

            # Hover to stabilize initial estimate
            time.sleep(3)

            print("\nMoving slowly to (2, 3, 0.8)...")

            # Slowly move to target center
            pc.go_to(X0, Y0, Z0)

            # Extra stabilization
            time.sleep(6)

            print("\nStarting circular motion around X-axis...")

            traj_points = generate_trajectory_points()

            follow_trajectory(pc, traj_points)

            print("\nReturning to center...")

            pc.go_to(X0, Y0, Z0)

            time.sleep(4)

            print("\nLanding...")

            pc.land()

    print("Mission complete")


# ==========================================
# ENTRY
# ==========================================

if __name__ == '__main__':
    run_mission()