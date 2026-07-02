# Simulation (ROS 2 + Gazebo + CrazySwarm)

## Overview

This directory contains the simulation environment developed for validating Crazyflie control algorithms before deploying them on physical hardware.

During the initial stages of development, frequent crashes caused motor failures, damaged propellers, and increased hardware downtime. To minimize these risks, a simulation-first workflow was adopted using **ROS 2**, **Gazebo**, and **CrazySwarm**, allowing algorithms to be tested and debugged in a safe virtual environment before real-world execution.

---

## Objectives

* Validate control algorithms without risking hardware.
* Develop and debug autonomous flight scripts safely.
* Simulate single and multi-drone scenarios.
* Test swarm behaviours before hardware deployment.
* Reduce development time caused by repeated crashes.

---

## Repository Structure

```text
simulation/
├── gazebo/
│   ├── simulation_ws/
│   ├── move_drone_ros_sim.py
│
├── ros2_swarm/
│   ├── ros2_ws/
│   ├── crazyflie_mapping_demo/
│
└── README.md
```

---

## Simulation Stack

The simulation environment is built using the following software components:

* **ROS 2** – Robot middleware for communication between simulation nodes.
* **Gazebo** – Physics-based simulator for modeling Crazyflie dynamics.
* **CrazySwarm** – Framework for controlling one or more simulated Crazyflies.
* **Python (cflib)** – High-level scripting interface for trajectory generation and experiment automation.

---

## Development Workflow

The general workflow followed throughout the project is illustrated below:

```text
Develop Algorithm
        │
        ▼
Implement in Python
        │
        ▼
Validate in Gazebo
        │
        ▼
Debug & Tune Parameters
        │
        ▼
Verify Stable Behaviour
        │
        ▼
Deploy on Real Crazyflie
```

---

## Simulated Experiments

The simulation environment was used to evaluate:

* Autonomous takeoff and landing
* Hover control
* Waypoint navigation
* Trajectory tracking
* Multi-drone coordination
* Swarm behaviour
* Communication and synchronization logic

---

## Why Simulation?

Simulation became an essential part of the development process after repeated hardware failures during experimental testing.

Some of the major issues encountered on the physical platform included:

* Motor damage caused by repeated crashes.
* Propeller failures.
* Localization errors leading to unstable flight.
* Incorrect controller parameters resulting in aggressive manoeuvres.
* Multi-drone synchronization issues.

By validating new algorithms in simulation first, these issues could be identified and corrected before deploying to the real Crazyflie, significantly reducing the likelihood of hardware damage.

---

## Advantages of the Simulation Workflow

* Safe environment for testing new algorithms.
* Faster debugging and development.
* Repeatable experiments under controlled conditions.
* Easy testing of multiple drones.
* Reduced hardware wear and maintenance.
* Improved confidence before real-world deployment.

---

## Future Improvements

Planned extensions to the simulation environment include:

* Dynamic obstacle avoidance.
* Vision-based localization.
* Formation control for larger drone swarms.
* Autonomous exploration and mapping.
* Reinforcement learning-based swarm navigation.
* Integration with SLAM and perception pipelines.

---

## Notes

The simulation environment was not intended to replace physical experiments but to complement them. All major algorithms were first verified in simulation and then successfully transferred to real Crazyflie hardware, resulting in a safer, more efficient, and systematic development process.
