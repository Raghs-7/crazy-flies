# Crazyflie Autonomous Missions

This repository contains the Bitcraze Crazyflie Python library integrated with autonomous flight missions, trajectory generation systems, and multi-UAV swarm experiments using the Loco Positioning System (LPS).

The project focuses on:

- Autonomous indoor navigation
- Trajectory planning and control
- Real-time telemetry logging
- Multi-UAV synchronized flight
- Swarm coordination in LPS environments
- Trajectory error analysis and evaluation

---

# Repository Structure

Development uses two separate repositories.

## 1. Crazyflie Client Repository

Used for:

- Running the Crazyflie desktop client
- Connecting to Crazyflie drones
- Monitoring console logs
- Verifying radio communication
- Checking estimator convergence
- Testing LPS positioning stability

Clone repository:

```bash
git clone https://github.com/bitcraze/crazyflie-clients-python.git
```

---

## 2. Crazyflie Python Library

Used for:

- Running Python-based autonomous missions
- Implementing flight trajectories
- Building swarm coordination systems
- Executing synchronized UAV operations
- Logging and evaluating trajectory performance

Clone repository:

```bash
git clone https://github.com/bitcraze/crazyflie-lib-python.git
```

---

# Implemented Flight Missions

The following autonomous missions and trajectory systems have been implemented successfully:

- Simple Autonomous Takeoff and Landing
- 3D Space Launch
- Square Path Trajectory
- Figure-8 Trajectory
- Circular Path Trajectory
- Dynamic Circular Swarm Motion
- Multi-UAV Simultaneous Flight
- Seeded Node Navigation System

---

# Swarm Flight System

The repository now supports simultaneous autonomous flight of two Crazyflie UAVs executing synchronized trajectories inside a shared LPS environment.

Current swarm capabilities include:

- Simultaneous takeoff and landing
- Synchronized trajectory execution
- Dynamic circular motion
- Phase-offset coordinated flight
- Independent flight altitudes
- Real-time position logging
- Trajectory visualization
- Mean Squared Error (MSE) path evaluation
- Safe multi-UAV separation

The current implementation supports:

| UAV | Flight Height |
|---|---|
| UAV 1 | 0.6 m |
| UAV 2 | 1.0 m |

Both UAVs execute synchronized circular trajectories while operating at different altitude layers for improved operational safety.

---

# Swarm Development Goals

Current swarm research focuses on:

- Stable synchronized motion
- Reliable multi-radio communication
- Collision-safe formation control
- Dynamic trajectory generation
- Real-time trajectory correction
- Scalable swarm architectures
- Robust indoor autonomous coordination

Future objectives include:

- Multi-UAV formation control
- Dynamic obstacle avoidance
- Decentralized swarm coordination
- Autonomous mission planning
- Real-time distributed navigation

---

# Current Mission: Seeded Node Navigator

The Seeded Node Navigator mission is designed for reliable autonomous navigation inside known LPS environments.

Instead of polling anchor coordinates dynamically through radio communication, the system loads predefined seeded coordinates from a local configuration file.

The UAV maintains a configurable safety buffer from anchors during navigation to avoid collisions with LPS infrastructure.

---

# Mission Workflow

During execution, the system performs the following sequence:

1. Initialize seeded coordinates for all LPS anchors
2. Load anchor configuration locally
3. Sort anchors by ID
4. Navigate sequentially between anchors
5. Maintain safety buffer distance
6. Hold constant hover altitude
7. Execute waypoint traversal
8. Navigate toward landing zone
9. Perform controlled vertical landing

---

# Autonomous Trajectory System

The repository includes trajectory systems for:

- Circular trajectories
- Figure-8 motion
- Square waypoint paths
- Dynamic center computation
- Phase-shifted synchronized motion
- Real-time path correction
- Multi-UAV coordinated flight

Trajectory accuracy is evaluated using:

- Real-time position logging
- Path deviation tracking
- Mean Squared Error (MSE)
- CSV telemetry logging
- Trajectory visualization plots

---

# System Requirements

Required hardware:

- Crazyflie 2.x
- Crazyradio PA
- Loco Positioning System (LPS)
- Positioning deck
- Fully charged batteries

Recommended for swarm experiments:

- Two Crazyradio adapters
- Dedicated radio channels per UAV
- Stable anchor placement
- Low-interference environment

---

# Environment Setup

## 1. Clone Repositories

### Crazyflie Client

```bash
git clone https://github.com/bitcraze/crazyflie-clients-python.git
```

### Crazyflie Python Library

```bash
git clone https://github.com/bitcraze/crazyflie-lib-python.git
```

---

# Running the Crazyflie Client

Navigate into the Crazyflie client repository:

```bash
cd crazyflie-clients-python
```

Activate the environment:

```bash
source env/bin/activate
```

Run the client:

```bash
cfclient
```

---

# Connecting the UAV

1. Connect the Crazyradio
2. Power the Crazyflie
3. Open the Crazyflie client
4. Connect to the drone
5. Verify estimator convergence
6. Verify LPS positioning stability
7. Check battery levels
8. Confirm radio communication

After all checks are completed, close the client terminal.

---

# Running Autonomous Missions

Open a second terminal and navigate into the Python library repository:

```bash
cd crazyflie-lib-python
```

Run any mission script:

```bash
python3 filename.py
```

Example:

```bash
python3 circle_test.py
```

---

# Example Swarm Mission

## Circular Swarm Motion

```bash
python3 swarm_circle.py
```

Mission features:

- Simultaneous multi-UAV flight
- Synchronized circular trajectories
- Independent flight heights
- Real-time trajectory logging
- MSE trajectory evaluation
- Dynamic trajectory generation

---

# Safety Notes

Before flight:

- Ensure the LPS system is active
- Verify anchor stability
- Confirm Crazyradio connectivity
- Check battery levels
- Wait for estimator convergence
- Clear the flight environment
- Verify radio configuration
- Test UAVs individually before swarm launch

For swarm operations:

- Maintain vertical separation
- Avoid overlapping startup positions
- Monitor radio packet stability
- Ensure synchronized estimator readiness

---

# Research and Development Focus

This project is currently focused on developing a modular indoor UAV swarm framework using Crazyflie platforms and LPS-based localization systems.

Primary research areas include:

- Autonomous indoor swarm navigation
- Real-time trajectory control
- Multi-UAV synchronization
- Precision path tracking
- Distributed swarm coordination
- Autonomous mission execution
- Indoor formation control
- Reliable swarm communication systems
