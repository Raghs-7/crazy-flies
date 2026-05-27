#!/usr/bin/env python3

import time
import math
import threading
import csv

import matplotlib.pyplot as plt

import cflib.crtp

from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander


# =========================================================
# DRONE URIS
# =========================================================

uris = [
    'radio://0/80/2M/E7E7E7E701',
    'radio://0/80/2M/E7E7E7E702'
]


# =========================================================
# GLOBALS
# =========================================================

positions = {}

trajectory_history = {
    uris[0]: [],
    uris[1]: []
}

# Stores instantaneous path errors
path_errors = {
    uris[0]: [],
    uris[1]: []
}

position_lock = threading.Lock()


# =========================================================
# FLIGHT PARAMETERS
# =========================================================

DT = 0.05

OMEGA = 0.5

MISSION_TIME = 25

MIN_RADIUS = 0.5

# Different heights for each drone
DRONE_HEIGHTS = {
    uris[0]: 0.6,
    uris[1]: 1.0
}


# =========================================================
# POSITION LOGGING
# =========================================================

def start_position_logging(scf):

    uri = scf.cf.link_uri

    log_conf = LogConfig(
        name=f'Position_{uri[-3:]}',
        period_in_ms=100
    )

    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')

    def log_callback(timestamp, data, logconf):

        x = data['stateEstimate.x']
        y = data['stateEstimate.y']
        z = data['stateEstimate.z']

        with position_lock:

            positions[uri] = (x, y, z)

            trajectory_history[uri].append((x, y))

    scf.cf.log.add_config(log_conf)

    log_conf.data_received_cb.add_callback(log_callback)

    log_conf.start()


# =========================================================
# COMPUTE DYNAMIC CIRCLE
# =========================================================

def compute_circle_parameters():

    while len(positions) < 2:
        time.sleep(0.1)

    uris_list = list(positions.keys())

    p1 = positions[uris_list[0]]
    p2 = positions[uris_list[1]]

    x1, y1, _ = p1
    x2, y2, _ = p2

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    radius = max(distance / 2.0, MIN_RADIUS)

    return center_x, center_y, radius


# =========================================================
# CIRCLE FLIGHT
# =========================================================

def circle_flight(scf, params):

    center_x, center_y, radius = params

    uri = scf.cf.link_uri

    flight_height = DRONE_HEIGHTS[uri]

    if uri.endswith('701'):
        phase_offset = 0
    else:
        phase_offset = math.pi

    try:

        with PositionHlCommander(
            scf,
            default_height=flight_height,
            controller=PositionHlCommander.CONTROLLER_PID
        ) as pc:

            print(f"\n{uri}")
            print(f"Taking off to {flight_height}m")

            time.sleep(3)

            start_x = center_x + radius * math.cos(phase_offset)
            start_y = center_y + radius * math.sin(phase_offset)

            print(f"{uri} moving to circle start")

            pc.go_to(start_x, start_y)

            time.sleep(4)

            print(f"{uri} starting smooth circular trajectory")

            theta = phase_offset

            start_time = time.time()

            while time.time() - start_time < MISSION_TIME:

                # =========================================
                # IDEAL CIRCLE POSITION
                # =========================================

                ideal_x = center_x + radius * math.cos(theta)
                ideal_y = center_y + radius * math.sin(theta)

                pc.go_to(ideal_x, ideal_y)

                # =========================================
                # COMPUTE TRACKING ERROR
                # =========================================

                with position_lock:

                    if uri in positions:

                        actual_x, actual_y, _ = positions[uri]

                        error = math.sqrt(
                            (ideal_x - actual_x) ** 2 +
                            (ideal_y - actual_y) ** 2
                        )

                        path_errors[uri].append(error)

                theta += OMEGA * DT

                time.sleep(DT)

            print(f"{uri} returning to center hover")

            pc.go_to(center_x, center_y)

            time.sleep(3)

            print(f"{uri} landing")

        print(f"{uri} landed safely")

    except Exception as e:

        print(f"{uri} ERROR: {e}")

        try:
            scf.cf.commander.send_stop_setpoint()
        except:
            pass


# =========================================================
# SAVE MSE RESULTS
# =========================================================

def save_mse_results():

    filename = "path_mse_results.csv"

    with open(filename, mode='w', newline='') as file:

        writer = csv.writer(file)

        writer.writerow([
            'drone',
            'samples',
            'mean_squared_error'
        ])

        for uri, errors in path_errors.items():

            if len(errors) == 0:
                continue

            mse = sum(e**2 for e in errors) / len(errors)

            writer.writerow([
                uri[-3:],
                len(errors),
                round(mse, 6)
            ])

            print(
                f"Drone {uri[-3:]} "
                f"MSE = {mse:.6f}"
            )

    print(f"\nMSE results saved to {filename}")


# =========================================================
# PLOT TRAJECTORIES
# =========================================================

def plot_trajectories(center_x, center_y, radius):

    plt.figure(figsize=(8, 8))

    for uri, points in trajectory_history.items():

        if len(points) == 0:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        label = uri[-3:]

        plt.plot(xs, ys, linewidth=2, label=f"Drone {label}")

        plt.scatter(xs[0], ys[0], s=100, marker='o')

        plt.scatter(xs[-1], ys[-1], s=100, marker='x')

    # =====================================================
    # IDEAL REFERENCE CIRCLE
    # =====================================================

    ideal_x = []
    ideal_y = []

    for t in range(360):

        theta = math.radians(t)

        ideal_x.append(center_x + radius * math.cos(theta))
        ideal_y.append(center_y + radius * math.sin(theta))

    plt.plot(
        ideal_x,
        ideal_y,
        linestyle='--',
        linewidth=1,
        label='Ideal Circle'
    )

    plt.scatter(
        center_x,
        center_y,
        s=120,
        marker='*',
        label='Circle Center'
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")

    plt.title("Crazyflie Swarm Circular Trajectories")

    plt.axis('equal')

    plt.grid(True)

    plt.legend()

    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    cflib.crtp.init_drivers()

    factory = CachedCfFactory(rw_cache='./cache')

    with Swarm(uris, factory=factory) as swarm:

        print("\nConnected to Crazyflies")

        print("\nWaiting for estimator convergence...")
        time.sleep(15)

        print("\nStarting position logging")

        swarm.parallel_safe(start_position_logging)

        time.sleep(5)

        print("\nDetected spawn positions:\n")

        for uri, pos in positions.items():

            print(
                f"{uri} -> "
                f"x={pos[0]:.2f}, "
                f"y={pos[1]:.2f}, "
                f"z={pos[2]:.2f}"
            )

        center_x, center_y, radius = compute_circle_parameters()

        print("\nComputed dynamic trajectory:\n")

        print(f"Circle Center : ({center_x:.2f}, {center_y:.2f})")
        print(f"Circle Radius : {radius:.2f}m")

        print("\nStarting synchronized circular swarm motion\n")

        params = (center_x, center_y, radius)

        swarm.parallel_safe(
            circle_flight,
            args_dict={
                uri: [params]
                for uri in uris
            }
        )

        print("\nMission complete\n")

        # =============================================
        # SAVE FINAL PATH ERROR METRICS
        # =============================================

        save_mse_results()

        print("\nPlotting trajectories...\n")

        plot_trajectories(center_x, center_y, radius)