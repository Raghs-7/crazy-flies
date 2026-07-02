# Crazyflie Autonomous Flight, Localization & Swarm Control

A comprehensive repository documenting the development of autonomous flight control, trajectory generation, localization, simulation, and multi-drone swarming using the **Bitcraze Crazyflie 2.x** platform.

This repository contains the complete workflow followed during the internship/project—from understanding the Crazyflie architecture and tuning the onboard controllers to implementing multi-drone coordination and validating algorithms in simulation before deploying them on hardware.

---

## Project Overview

The primary objective of this project was to understand the complete software and hardware stack of the Crazyflie ecosystem and progressively develop reliable autonomous flight capabilities.

The work includes:

* Understanding the Crazyflie firmware architecture
* Studying the Position High-Level Commander
* Investigating the Loco Positioning System (LPS)
* Learning Two-Way Ranging (TWR) localization
* PID tuning for stable autonomous flight
* Single-drone trajectory generation
* Multi-drone coordination using multithreading
* Boids-based swarm behaviour
* ROS 2 & Gazebo simulation
* CrazySwarm integration
* Localization optimization through anchor placement

---

# Repository Structure

```text
.
├── single_drone/
│   ├── circle_trajectory/
│   └── figure8_trajectory/
│
├── multi_drone/
│   ├── two_drone_circle/
│   └── swarming/
│
├── simulation/
│   ├── gazebo/
│   └── ros2_swarm/
│
├── launch/
│
└── README.md
```

---

# Features

## Autonomous Flight

* Takeoff & Landing
* Hover Control
* Position Control
* Waypoint Navigation

---

## Trajectory Generation

Implemented trajectories include:

* Circular Trajectory
* Figure-8 Trajectory
* Multi-drone Circular Flight

Trajectory plots and logs are included with each implementation.

---

## Localization

The project investigates the complete Loco Positioning System including:

* Two-Way Ranging (TWR)
* Anchor configuration
* Position estimation
* Kalman Filter based localization
* Anchor placement optimization

Several localization issues encountered during experimentation are documented together with the corresponding solutions.

---

## PID Tuning

Extensive PID tuning was performed using the Crazyflie Client.

Topics covered include:

* Effect of Proportional Gain
* Integral Windup
* Derivative Damping
* Oscillation Reduction
* Trajectory Tracking Improvement

Both pre- and post-tuning flight behaviour are documented.

---

## Multi-Drone Control

Implementation of concurrent control for multiple Crazyflies using Python multithreading.

Features include:

* Independent communication links
* Synchronized takeoff
* Parallel trajectory execution
* Two-drone circular flight

---

## Swarm Behaviour

Implementation of a Boids-inspired swarm algorithm incorporating:

* Separation
* Alignment
* Cohesion

The repository includes both simulation and hardware experiments.

---

## Simulation

Before deployment on real hardware, all algorithms were validated in simulation.

Simulation stack:

* ROS 2
* Gazebo
* CrazySwarm

This significantly reduced hardware failures and enabled safe debugging.

---

# Challenges Faced

Throughout the project several practical challenges were encountered, including:

* High localization variance
* Motor failures due to repeated crashes
* TWR communication issues
* Anchor placement inaccuracies
* PID instability
* Integral windup
* Multi-drone synchronization
* Communication latency

Each challenge is documented together with the engineering approach used to resolve it.

---

# Technologies Used

### Hardware

* Bitcraze Crazyflie 2.x
* Loco Positioning System
* Loco Anchors
* Crazyradio PA

### Software

* Python
* cflib
* Crazyflie Client (CFClient)
* ROS 2
* Gazebo
* CrazySwarm
* Ubuntu Linux

---

# Experimental Workflow

```
Understand Crazyflie
          │
          ▼
Learn Position Commander
          │
          ▼
Single Drone Flights
          │
          ▼
PID Tuning
          │
          ▼
Trajectory Tracking
          │
          ▼
Localization Optimization
          │
          ▼
Simulation using ROS2 + Gazebo
          │
          ▼
Multi-Drone Control
          │
          ▼
Swarm Behaviour
```

---

# Future Work

* Obstacle Avoidance
* Formation Control
* Vision-Based Localization
* SLAM Integration
* Dynamic Path Planning
* Large Scale Swarms
* Reinforcement Learning for Swarm Navigation

---

# Acknowledgements

This work was carried out as part of an internship/project focused on autonomous aerial robotics using the Crazyflie platform. The project builds upon the open-source ecosystem developed by Bitcraze and the robotics community.

---

## License

This repository is intended for educational and research purposes.

