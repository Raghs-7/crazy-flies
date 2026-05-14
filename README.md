```markdown
# Crazyflie Autonomous Room Navigation

This project implements an autonomous flight sequence for the Bitcraze Crazyflie 2.1 using the **Loco Positioning System (LPS)**. Unlike basic optical flow navigation, this script utilizes absolute room coordinates, allowing the drone to be deployed from any starting position within the LPS-active area.

## 🚁 Mission Overview
The script executes a targeted navigation sequence:
1.  **Autonomous Takeoff**: The drone initializes at any random location within the anchor constellation.
2.  **Origin Calibration**: Immediately navigates to the physical room origin $(0, 0, 0.5)$.
3.  **Waypoint Alpha**: Travels to $(0, 2, 0.5)$ and hovers for a steady state.
4.  **Recenter**: Returns to $(0, 0, 0.5)$ to verify positional integrity.
5.  **Waypoint Bravo**: Travels to $(0, 4, 0.5)$ and hovers.
6.  **RTL (Return to Launch)**: Returns to the origin $(0, 0, 0.5)$ and performs a controlled landing.

## 🛠 Prerequisites
* **Hardware**: 
    * Crazyflie 2.1
    * Loco Positioning Deck
    * Loco Positioning Nodes (configured in TWR or TDOA mode)
    * Crazyradio PA
* **Software**: 
    * Python 3.10+
    * `cflib` (Crazyflie Python Library)

## 🚀 Getting Started

### 1. Installation
Ensure you have the Crazyflie library installed. From the root of this repository, run:
```bash
pip install .

```

### 2. Configuration

Open the main script and ensure the `URI` matches your Crazyflie's specific radio address:

```python
URI = 'radio://0/80/2M/E7E7E7E7E7' 

```

### 3. Execution

Place the drone anywhere in the flight arena, ensuring the Loco Nodes are powered on and the drone has a position lock (indicated by a steady green M4 LED), then run:

```bash
python3 run_mission.py

```

## ⚠️ Safety Information

* **Emergency Stop**: Keep your hand near the keyboard. Press `Ctrl+C` in the terminal to immediately kill the motors and terminate the script.
* **Space Requirement**: This mission requires at least **4 meters** of clear space along the Y-axis. Ensure your LPS anchors are calibrated accurately before flight.
* **Battery**: High-level commands require stable voltage. Do not attempt autonomous missions if the battery is below 3.7V.

---

**Collaborators**: Raghav, Anika and Dewang


```

```
