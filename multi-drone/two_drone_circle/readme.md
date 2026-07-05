# Multi-Drone Circular Trajectory

## Overview

This module demonstrates synchronized circular trajectory tracking using two Crazyflie drones. Each drone is controlled independently through separate communication links while maintaining coordinated flight using Python multithreading.

The experiment serves as a foundation for cooperative multi-robot control and swarm applications.

---

## Objectives

* Simultaneously control two Crazyflies.
* Execute synchronized circular trajectories.
* Develop multithreaded drone control using cflib.
* Validate communication reliability and synchronization.

---

## Repository Contents

```text
two_drone_circle/
├── two_cf_circle_path.py
├── trajectory_plots/
├── logs/
└── README.md
```

---

## Working Principle

Each Crazyflie is assigned an independent communication thread responsible for transmitting position commands.

The program performs:

* Connection establishment
* Synchronized takeoff
* Parallel trajectory execution
* Coordinated landing

The onboard position controllers execute the received position commands independently.

---
## Demonstration Video

[Watch the Dual Drone Circular Flight Demo](https://drive.google.com/file/d/15mfLE9bK-xCZH9vjKYqjsXOArD6pt6ze/view?usp=drive_link)
## Challenges

During development, several issues were encountered:

* TWR localization conflicts
* Communication synchronization
* Position estimation errors
* Anchor placement optimization

These issues were addressed by improving anchor placement, updating localization parameters, and refining the multithreaded implementation.

---

## Results

Both drones successfully completed synchronized circular trajectories while maintaining safe separation and stable localization.

This implementation forms the basis for more advanced swarm behaviours.
