# Single Drone Circular Trajectory

## Overview

This module demonstrates autonomous circular trajectory tracking using a single Crazyflie 2.x drone. The trajectory is executed using the Python cflib High-Level Commander interface, which generates position setpoints while the onboard controller handles low-level stabilization.

The objective of this experiment was to understand waypoint-based trajectory generation, evaluate the onboard position controller, and perform PID tuning to achieve smooth and accurate path tracking.

---

## Objectives

* Generate a circular trajectory in the horizontal plane.
* Study the behaviour of the Position High-Level Commander.
* Analyze tracking accuracy.
* Tune PID gains to reduce oscillations and overshoot.
* Record and visualize the executed trajectory.

---

## Repository Contents

```text
circle_trajectory/
├── circle_path.py
├── circle_velocity_control.py
├── circle_trajectory_log.csv
├── circle_trajectory_3d.png
├── circle_trajectory_xy_topdown.png
└── README.md
```

---

## Working Principle

The script computes successive points along a circular path using parametric equations and continuously sends these position setpoints to the Crazyflie.

The onboard controller then performs:

Position Control → Velocity Control → Attitude Control → Motor Commands

No direct motor commands are issued from Python.

---

## Results

The drone successfully completed circular trajectories while maintaining stable altitude. Multiple iterations of PID tuning were performed to improve tracking accuracy and reduce oscillations.

Trajectory plots are included for visual analysis.

---

## Future Improvements

* Variable-radius trajectories
* Helical (3D) trajectories
* Adaptive velocity profiles
* Obstacle avoidance during circular flight
