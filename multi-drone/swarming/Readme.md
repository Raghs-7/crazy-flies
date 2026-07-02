# Swarm Behaviour using Boids Algorithm

## Overview

This module implements decentralized swarm behaviour for multiple Crazyflie drones using the Boids algorithm. The objective is to enable autonomous collective motion through simple local interaction rules without centralized path planning.

The implementation was first validated in simulation before being deployed on physical hardware.

---

## Objectives

* Understand decentralized swarm intelligence.
* Implement Boids-based collective behaviour.
* Validate swarm algorithms in simulation.
* Deploy the algorithm on Crazyflie drones.

---

## Repository Contents

```text
swarming/
├── real_boids_swarm.py
├── plots/
├── logs/
└── README.md
```

---

## Swarm Rules

The Boids algorithm is based on three fundamental behaviours:

### Separation

Avoid collisions with neighbouring drones.

### Alignment

Match the heading and velocity of nearby drones.

### Cohesion

Move towards the local center of neighbouring drones.

The combination of these simple behaviours produces complex and coordinated swarm motion.

---

## Development Process

The swarm controller was initially developed and tested in Gazebo using ROS 2 and CrazySwarm to minimize hardware risks. After validating the behaviour in simulation, the implementation was transferred to the real Crazyflie platform.

---

## Results

The implemented swarm algorithm successfully demonstrated coordinated collective motion while maintaining collision avoidance and stable inter-drone spacing.

This work provides the foundation for future formation control and large-scale autonomous swarms.
