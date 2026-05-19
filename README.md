# Crazyflie Autonomous Missions

This repository contains the Bitcraze Crazyflie Python library integrated with custom autonomous flight sequences for Loco Positioning System (LPS) environments.

## Repository Setup

To work with the Crazyflie drone, we use two separate repositories:

1. The Crazyflie Client repository for:
   - Running the Crazyflie client
   - Accessing the console
   - Connecting and testing the drone manually

2. The `crazyflie-lib-python` repository for:
   - Running Python scripts
   - Creating autonomous flight patterns
   - Implementing trajectories and missions

---

## Implemented Flight Missions

Till now, the following flight missions and trajectories have been implemented:

- 3D Space Launch
- Simple Launch
- Square Path Trajectory
- Figure-8 Trajectory
- Circle Path Trajectory

---

## Current Mission: Seeded Node Navigator

This mission is designed for reliable navigation inside a known LPS environment. Instead of polling anchor positions through radio communication, the script uses predefined seeded coordinates for navigation.

The drone also maintains a safety buffer from all anchors during flight to avoid collisions with the LPS infrastructure.

### Mission Sequence

Upon execution, the drone will:

1. Initialize seeded coordinates for all LPS anchors from the local configuration file.
2. Sort anchors by ID and visit them sequentially.
3. Maintain a safety buffer distance while navigating near anchors.
4. Maintain a constant hover height during flight.
5. Navigate to the designated landing zone.
6. Perform a controlled vertical landing.

---

# How to Run

## 1. Clone Required Repositories

Clone both repositories:

### Crazyflie Client
```bash
git clone https://github.com/bitcraze/crazyflie-clients-python.git
```

### Crazyflie Python Library
```bash
git clone https://github.com/bitcraze/crazyflie-lib-python.git
```

---

## 2. Run the Crazyflie Client

Open the terminal inside the Crazyflie Client repository and run:

```bash
source env/bin/activate
cfclient
```

Connect the drone using the Crazyflie client.

After the connection and required checks are completed, close the client terminal.

---

## 3. Run Python Flight Scripts

Open a second terminal and navigate to the `crazyflie-lib-python` repository.

Run the required Python script using:

```bash
python3 filename.py
```

Example:

```bash
python3 circle_test.py
```

---

## Notes

- Ensure the Loco Positioning System (LPS) is active before running autonomous missions.
- Make sure the Crazyradio is connected properly.
- Verify battery levels before flight.
- Run scripts only after successfully connecting through the Crazyflie client.
