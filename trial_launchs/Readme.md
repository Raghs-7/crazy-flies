# Launch Scripts

## Overview

This directory contains utility launch scripts used throughout the project for initializing different Crazyflie experiments.

These scripts automate common tasks such as establishing communication, configuring flight parameters, and launching predefined experiments.

---

## Available Scripts

### `simple_launch.py`

Launches a basic single-drone experiment for testing communication and autonomous flight.

---

### `square_launch.py`

Launches a square trajectory experiment used during the initial stages of trajectory generation and controller evaluation.

---

### `3dspace_launch.py`

Launches experiments involving three-dimensional waypoint navigation and spatial trajectory execution.

---

### `two_cf_launch_without_ros.py`

Launch script for establishing communication with two Crazyflies using cflib without relying on the ROS 2 framework. This script was primarily used during the early stages of multi-drone development and debugging.

---

## Purpose

These launch scripts simplify experiment execution by reducing repetitive setup steps and ensuring consistent initialization across different flight tests.

They also serve as reusable entry points for future trajectory and swarm experiments.
