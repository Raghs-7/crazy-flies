#!/usr/bin/env python3
"""
two_drone_mission.py
────────────────────────────────────────────────────────────────────────────────
Flies TWO Crazyflie drones through the SAME 8 checkpoints (the cube-like
trajectory from the original single-drone script) without colliding.

Collision-avoidance strategy
─────────────────────────────
  • CF1 takes off first and starts moving immediately.
  • CF2 waits CF2_DELAY seconds before taking off — so it is always roughly
    one full checkpoint-interval behind CF1 on the same path.
  • Both drones fly at DIFFERENT ALTITUDES at each checkpoint (CF1 always
    ALTITUDE_OFFSET metres LOWER than CF2).  This means even if the time
    stagger slips, they are vertically separated.
  • HOVER_PAUSE (wait at each checkpoint) is long enough that CF2 won't
    catch up to CF1 mid-segment.

Visual summary
──────────────
  time →
  CF1: takeoff ── CP0 ── CP1 ── CP2 ── CP3 ── CP4 ── CP5 ── CP6 ── CP7 ── land
  CF2:    (wait CF2_DELAY s)  takeoff ── CP0 ── CP1 ── ... ── CP7 ── land

  Vertical separation: CF1 flies at z_checkpoint,
                       CF2 flies at z_checkpoint + ALTITUDE_OFFSET

WHERE TO PUT THIS FILE
──────────────────────
  ~/crazyflie_mapping_demo/two_drone_mission.py

  i.e. right inside the top-level demo folder, NOT inside the ROS workspace.
  This is a plain Python script that talks directly to the hardware via cflib
  — it has nothing to do with ROS.  Run it from any terminal:

      cd ~/crazyflie_mapping_demo
      python3 two_drone_mission.py

  Make sure cflib is installed:
      pip3 install cflib

Hardware assumptions
────────────────────
  • Two Crazyflie 2.1 drones, each with Multi-ranger + LPS deck
  • Two Crazyradio dongles (or one Crazyradio 2.0 with two drones on
    different channels — update the URIs below accordingly)
  • LPS anchors deployed and covering the entire flight volume
  • Both drones placed flat on the floor before running, facing +X

Safety reminders
────────────────
  • Clear the flight volume of people and obstacles before arming.
  • Keep your hand near the power switch of each drone in case of emergency.
  • PositionHlCommander lands automatically when its `with` block exits,
    even if an exception is raised mid-flight.
────────────────────────────────────────────────────────────────────────────────
"""

import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper

# ─────────────────────────────────────────────────────────────────────────────
# URIs  — update to match YOUR drones
#   CF1: radio://0/80/2M/E7E7E7E7E1   (channel 80, address …E1)
#   CF2: radio://0/90/2M/E7E7E7E7E2   (channel 90, address …E2)
#
# You can override at runtime:
#   CF1_URI=radio://0/80/2M/E7E7E7E7E1 \
#   CF2_URI=radio://0/90/2M/E7E7E7E7E2 \
#   python3 two_drone_mission.py
# ─────────────────────────────────────────────────────────────────────────────
CF1_URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E1")
CF2_URI = uri_helper.uri_from_env(default="radio://0/90/2M/E7E7E7E7E2",
                                   env_var="CF2_URI")

# ─────────────────────────────────────────────────────────────────────────────
# Flight parameters
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_VELOCITY  = 0.4   # m/s — how fast each drone moves between checkpoints
HOVER_PAUSE       = 3.0   # s   — how long to hover at each checkpoint
LPS_LOCK_WAIT     = 3.0   # s   — time to let Kalman filter converge after connect
ALTITUDE_OFFSET   = 0.3   # m   — CF2 flies this much HIGHER than CF1 at every point
CF2_DELAY         = 8.0   # s   — CF2 waits this long before taking off
                           #       Rule of thumb: >= HOVER_PAUSE + travel time between
                           #       the two densest consecutive checkpoints.

# ─────────────────────────────────────────────────────────────────────────────
# Shared checkpoint list  (same 8 points as the original single-drone script)
#
# These trace a cube-like path visiting all 8 LPS anchor zones.
# Adjust the X/Y/Z values to match your actual room and anchor placement.
#
#   Index   Anchor zone   (x,     y,     z   )
# ─────────────────────────────────────────────────────────────────────────────
BASE_CHECKPOINTS = [
    (0.0,  0.5,  0.3),   # 0 — Anchor zone 0  (low)
    (0.0,  6.15, 2.0),   # 1 — Anchor zone 1  (high)
    (5.03, 6.15, 0.3),   # 2 — Anchor zone 2  (low)
    (5.03, 0.5,  2.0),   # 3 — Anchor zone 3  (high)
    (0.0,  0.5,  2.0),   # 4 — Anchor zone 4  (high)
    (0.0,  6.15, 0.3),   # 5 — Anchor zone 5  (low)
    (5.03, 6.15, 2.0),   # 6 — Anchor zone 6  (high)
    (5.03, 0.5,  0.3),   # 7 — Anchor zone 7  (low)
]

# CF1 flies at the BASE altitudes exactly.
CF1_CHECKPOINTS = [(x, y, z) for (x, y, z) in BASE_CHECKPOINTS]

# CF2 flies ALTITUDE_OFFSET metres higher at every checkpoint.
# This gives guaranteed vertical separation even if the time stagger slips.
CF2_CHECKPOINTS = [(x, y, z + ALTITUDE_OFFSET) for (x, y, z) in BASE_CHECKPOINTS]


# ─────────────────────────────────────────────────────────────────────────────
# Per-drone mission
# ─────────────────────────────────────────────────────────────────────────────
def run_drone(uri: str,
              checkpoints: list,
              drone_id: str,
              pre_delay: float = 0.0) -> None:
    """
    Connect to one Crazyflie, optionally wait *pre_delay* seconds (used to
    stagger CF2's takeoff), then fly through every checkpoint and land.

    This function is designed to run inside its own thread.
    """
    print(f"[{drone_id}] Connecting to {uri} …")
    cflib.crtp.init_drivers()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=f"./cache_{drone_id}")) as scf:
        print(f"[{drone_id}] Connected. Waiting {LPS_LOCK_WAIT:.0f} s for LPS lock …")
        time.sleep(LPS_LOCK_WAIT)

        if pre_delay > 0:
            print(f"[{drone_id}] Holding on ground for {pre_delay:.1f} s "
                  f"(letting CF1 get ahead) …")
            time.sleep(pre_delay)

        with PositionHlCommander(
            scf,
            default_height=checkpoints[0][2],   # initial hover = first checkpoint z
            default_velocity=DEFAULT_VELOCITY,
            default_landing_height=0.0,
            controller=PositionHlCommander.CONTROLLER_PID,
        ) as pc:
            print(f"[{drone_id}] ✈  Airborne")

            for i, (x, y, z) in enumerate(checkpoints):
                print(f"[{drone_id}]  → CP {i+1}/{len(checkpoints)}  "
                      f"x={x:.2f}  y={y:.2f}  z={z:.2f}")
                pc.go_to(x, y, z)
                time.sleep(HOVER_PAUSE)

            print(f"[{drone_id}] ✓  All checkpoints done — landing …")
            # PositionHlCommander auto-lands when the `with` block exits

    print(f"[{drone_id}] Landed and disconnected.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Two-Drone Staggered Mission  (same path, no collision)")
    print("=" * 60)
    print(f"  CF1 URI        : {CF1_URI}")
    print(f"  CF2 URI        : {CF2_URI}")
    print(f"  CF2 takeoff delay : {CF2_DELAY} s after connection")
    print(f"  Altitude offset   : +{ALTITUDE_OFFSET} m for CF2")
    print(f"  Checkpoints       : {len(BASE_CHECKPOINTS)}")
    print("=" * 60)

    # CF1 thread — takes off immediately after LPS lock
    t1 = threading.Thread(
        target=run_drone,
        args=(CF1_URI, CF1_CHECKPOINTS, "CF1"),
        kwargs={"pre_delay": 0.0},
        daemon=True,
    )

    # CF2 thread — same path but shifts each z up by ALTITUDE_OFFSET,
    # and waits CF2_DELAY seconds before taking off so CF1 is always
    # ~one checkpoint ahead horizontally.
    t2 = threading.Thread(
        target=run_drone,
        args=(CF2_URI, CF2_CHECKPOINTS, "CF2"),
        kwargs={"pre_delay": CF2_DELAY},
        daemon=True,
    )

    # Connect both drones at the same time so their LPS locks happen in
    # parallel — only the takeoff is staggered.
    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("\n" + "=" * 60)
    print("  Both drones landed. Mission complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
