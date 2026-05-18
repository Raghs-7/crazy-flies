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

# -------- Circular Trajectory --------

r = 0.5

trajectory = [
    CompressedStart(1.0, 3.0, 0.8, 0.0),

    # quarter 1
    CompressedSegment(
        2.0,
        [0.0, r, 0.0],
        [-r, 0.0, 0.0],
        [],
        []
    ),

    # quarter 2
    CompressedSegment(
        2.0,
        [-r, 0.0, 0.0],
        [0.0, -r, 0.0],
        [],
        []
    ),

    # quarter 3
    CompressedSegment(
        2.0,
        [0.0, -r, 0.0],
        [r, 0.0, 0.0],
        [],
        []
    ),

    # quarter 4
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

    # XY
    plt.figure()
    plt.plot(ax, ay, label='Actual')
    plt.plot(ex, ey, '--', label='Expected')
    plt.title("Top-down XY")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("trajectory_xy_topdown.png")
    print("Saved trajectory_xy_topdown.png")

    # X(t)
    plt.figure()
    plt.plot(t, ax)
    plt.plot(t, ex, '--')
    plt.title("X over time")
    plt.grid(True)
    plt.savefig("trajectory_x.png")
    print("Saved trajectory_x.png")

    # Y(t)
    plt.figure()
    plt.plot(t, ay)
    plt.plot(t, ey, '--')
    plt.title("Y over time")
    plt.grid(True)
    plt.savefig("trajectory_y.png")
    print("Saved trajectory_y.png")

    # Z(t)
    plt.figure()
    plt.plot(t, az)
    plt.plot(t, ez, '--')
    plt.title("Z over time")
    plt.grid(True)
    plt.savefig("trajectory_z.png")
    print("Saved trajectory_z.png")

    # 3D
    fig = plt.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.plot(ax, ay, az)
    ax3.plot(ex, ey, ez, '--')
    ax3.set_title("3D Trajectory")
    plt.show()

# -------- RUN SEQUENCE ----------

def run_sequence(cf, trajectory_id, duration):
    commander = cf.high_level_commander

    log_conf.start()

    cf.platform.send_arming_request(True)
    time.sleep(1.0)

    # Takeoff
    commander.takeoff(1.0, 2.0)
    time.sleep(3)

    # Move safely to trajectory start point
    commander.go_to(1.0, 3.0, 0.8, 0, 4.0, relative=False)
    time.sleep(4.2)

    # Start circular compressed trajectory
    commander.start_trajectory(trajectory_id, 1.0, True)
    time.sleep(duration)

    # Move to landing point
    LAND_X = 2.50
    LAND_Y = 3.36
    SAFE_Z = 0.5

    commander.go_to(LAND_X, LAND_Y, SAFE_Z, 0, 3, relative=False)
    time.sleep(3.2)

    # Land
    commander.land(0.0, 3.0)
    time.sleep(3.2)

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

    # ---- SAVE CSV ----
    save_log_to_csv(log_data_list)

    # ---- PLOT ----
    plot_and_save_data(log_data_list)