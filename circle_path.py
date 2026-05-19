#!/usr/bin/env python3

import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import CompressedSegment, CompressedStart, MemoryElement
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

# URI to the Crazyflie
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E2')

# -------- First Octant Circular Trajectory --------
# Circle Center will be at local relative (0.5, 0.5). Radius = 0.35.
# This means the absolute closest it gets to any boundary is 0.15m (safely positive!).
r = 0.35

trajectory = [
    # Start point of the circle relative to our new positive offset waypoint
    # We start at CenterX + Radius (0.5 + 0.35 = 0.85), CenterY (0.5)
    CompressedStart(0.85, 0.5, 0.6, 0.0),

    # Quarter 1 loop segment
    CompressedSegment(
        2.0,
        [0.0, r, 0.0],
        [-r, 0.0, 0.0],
        [],
        []
    ),

    # Quarter 2 loop segment
    CompressedSegment(
        2.0,
        [-r, 0.0, 0.0],
        [0.0, -r, 0.0],
        [],
        []
    ),

    # Quarter 3 loop segment
    CompressedSegment(
        2.0,
        [0.0, -r, 0.0],
        [r, 0.0, 0.0],
        [],
        []
    ),

    # Quarter 4 loop segment
    CompressedSegment(
        2.0,
        [r, 0.0, 0.0],
        [0.0, r, 0.0],
        [],
        []
    ),
]

# Global variable for logs
log_data_list = []
log_conf = None


def upload_trajectory(cf, trajectory_id, trajectory):
    trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
    trajectory_mem.trajectory = trajectory

    if not trajectory_mem.write_data_sync():
        print('Upload failed!')
        sys.exit(1)

    cf.high_level_commander.define_trajectory(
        trajectory_id, 0, len(trajectory),
        type=HighLevelCommander.TRAJECTORY_TYPE_POLY4D_COMPRESSED
    )

    total_duration = 0
    for seg in trajectory[1:]:
        total_duration += seg.duration
    return total_duration


# -------- LOG CALLBACK ----------

def log_callback(timestamp, data, logconf):
    log_data_list.append({
        'time': timestamp,
        'actual_x': data.get('kalman.stateX', 0),
        'actual_y': data.get('kalman.stateY', 0),
        'actual_z': data.get('kalman.stateZ', 0),
        'expected_x': data.get('ctrltarget.x', 0),
        'expected_y': data.get('ctrltarget.y', 0),
        'expected_z': data.get('ctrltarget.z', 0)
    })


# -------- LOG SETUP ----------

def setup_logging(cf):
    global log_conf

    log_conf = LogConfig(name='TrajectoryLog', period_in_ms=100)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    log_conf.add_variable('ctrltarget.x', 'float')
    log_conf.add_variable('ctrltarget.y', 'float')
    log_conf.add_variable('ctrltarget.z', 'float')

    cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(log_callback)


# -------- SAVE CSV ----------

def save_log_to_csv(log_data, filename='trajectory_log.csv'):
    if not log_data:
        print("No logs to save!")
        return

    keys = ['time', 'actual_x', 'actual_y', 'actual_z',
            'expected_x', 'expected_y', 'expected_z']

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in log_data:
            writer.writerow(row)

    print(f"Saved CSV: {filename}")


# -------- PLOTTING ----------

def plot_and_save_data(log_data):
    if not log_data:
        print('No data to plot.')
        return

    t = np.array([d['time'] for d in log_data]) / 1000.0
    t -= t[0]

    ax = np.array([d['actual_x'] for d in log_data])
    ay = np.array([d['actual_y'] for d in log_data])
    az = np.array([d['actual_z'] for d in log_data])

    ex = np.array([d['expected_x'] for d in log_data])
    ey = np.array([d['expected_y'] for d in log_data])
    ez = np.array([d['expected_z'] for d in log_data])

    # XY Plot
    plt.figure()
    plt.plot(ax, ay, label='Actual')
    plt.plot(ex, ey, '--', label='Expected')
    plt.title("Top-down XY (Should be entirely positive)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("trajectory_xy_topdown.png")
    print("Saved trajectory_xy_topdown.png")

    # 3D
    fig = plt.figure()
    fig.add_subplot(111, projection='3d').plot(ax, ay, az, label='Actual')
    plt.savefig("trajectory_3d.png")


# -------- RUN SEQUENCE ----------

def run_sequence(cf, trajectory_id, duration):
    commander = cf.high_level_commander

    log_conf.start()

    cf.platform.send_arming_request(True)
    time.sleep(1.0)

    # 1. Takeoff vertically relative to wherever it sits right now
    print("Taking off...")
    commander.takeoff(0.6, 2.5)
    time.sleep(3.5)

    # 2. Cruise deep into the local first octant relative to home (x > 0, y > 0)
    print("Moving into local positive space...")
    commander.go_to(0.5, 0.5, 0.6, yaw=0, duration=4.0, relative=True)
    time.sleep(4.5)

    # 3. Fire off the circular trajectory array built in positive bounds
    print("Executing circle...")
    commander.start_trajectory(trajectory_id, time_scale=1.0, relative=True)
    time.sleep(duration + 1.0)

    # 4. Bring it straight down safely at its relative endpoint
    print("Landing...")
    commander.land(0.0, duration=3.0)
    time.sleep(3.5)

    commander.stop()
    log_conf.stop()


# -------- MAIN ----------

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        setup_logging(cf)
        traj_id = 1

        duration = upload_trajectory(cf, traj_id, trajectory)
        print(f"Trajectory duration: {duration:.1f}s")

        reset_estimator(cf)
        run_sequence(cf, traj_id, duration)

    # ---- SAVE AND PLOT ----
    save_log_to_csv(log_data_list)
    plot_and_save_data(log_data_list)
