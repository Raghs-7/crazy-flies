import time
import cflib.crtp

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper

# -----------------------------
# RADIO URI
# -----------------------------
URI = uri_helper.uri_from_env(
    default='radio://0/80/2M/E7E7E7E7E2'
)

# -----------------------------
# CHECKPOINTS (LPS Anchor Areas)
# Define coordinates near anchors
# Modify these according to YOUR room
# -----------------------------

CHECKPOINTS = [
    (0.0, 0.5, 0.3),   # Anchor zone 0
    (0.0, 6.15, 2.0),   # Anchor zone 1
    (5.03, 6.15, 0.3),   # Anchor zone 2
    (5.03, 0.5, 2.0),   # Anchor zone 3

    (0.0, 0.5, 2.0),   # Anchor zone 4
    (0.0, 6.15, 0.3),   # Anchor zone 5
    (5.03, 6.15, 2.0),   # Anchor zone 6
    (5.03, 0.5, 0.3),   # Anchor zone 7
]

# This creates a cube-like trajectory


def fly_to_checkpoints(pc):
    """
    Fly through all anchor checkpoints
    """

    for i, point in enumerate(CHECKPOINTS):

        x, y, z = point

        print("-" * 40)
        print(f"Going to checkpoint {i+1}")
        print(f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        print("-" * 40)

        pc.go_to(x, y, z)

        # Wait for stabilization
        time.sleep(3)


def run_mission():

    print("Initializing CRTP drivers...")
    cflib.crtp.init_drivers()

    with SyncCrazyflie(
        URI,
        cf=Crazyflie(rw_cache='./cache')
    ) as scf:

        print("Connected to Crazyflie")
        print("Waiting for LPS positioning...")
        time.sleep(3)

        # High-level commander
        with PositionHlCommander(
            scf,
            default_height=0.5,
            default_velocity=0.4,
            default_landing_height=0.0
        ) as pc:

            print("Takeoff complete")

            # Fly through checkpoints
            fly_to_checkpoints(pc)

            print("Returning to start point...")
            pc.go_to(0.0, 2.0, 0.5)
            time.sleep(2)

            print("Landing...")
            pc.land()

    print("Mission completed.")


if __name__ == '__main__':
    run_mission()
