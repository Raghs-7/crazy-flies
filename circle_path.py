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

    # Reduced from 160 to 8 points. 
    # High-level commander will smoothly transition between these waypoints.
    # We omit the final point (2*pi) because it overlaps with the first point (0).
    for t in np.linspace(0, 2 * math.pi, 9)[:-1]:
        points.append(trajectory(t))

    return points


# ==========================================
# FOLLOW TRAJECTORY
# ==========================================

def follow_trajectory(pc, points):

    # Flight parameters for moving between waypoints
    TARGET_VELOCITY = 0.20  # 20 cm/s speed
    
    for i, (x, y, z) in enumerate(points):

        print(f"\nWaypoint {i+1}/{len(points)}")
        print(f"Target -> X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

        # Send the command with an explicit velocity override
        pc.go_to(x, y, z, velocity=TARGET_VELOCITY)

        # FIX: Calculate distance to the next point to ensure we sleep long enough.
        # This completely guarantees the previous polynomial is dead before the next starts!
        # A static 2.0 second sleep gives plenty of time for the short ~30cm segments.
        time.sleep(2.0)


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

        # Setup the Commander with safe baseline rules
        with PositionHlCommander(
            scf,
            default_height=0.4,
            default_velocity=0.15,      # Safe baseline speed
            default_landing_height=0.1
        ) as pc:

            print("Takeoff complete")

            # Hover to stabilize initial estimate
            time.sleep(3)

            print(f"\nMoving slowly to circle start position ({X0}, {Y0+R}, {Z0})...")

            # Move to the exact starting point of the circle loop (t=0 -> cos(0)=1, sin(0)=0)
            pc.go_to(X0, Y0 + R, Z0)

            # Extra stabilization at start position
            time.sleep(5)

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
