# Crazyflie Autonomous Missions 

This repository contains the Bitcraze Crazyflie python library integrated with custom autonomous flight sequences, specifically designed for Loco Positioning System (LPS) environments.

## Current Mission: Seeded Node Navigator
This mission is designed for high-reliability navigation within a known LPS environment. Instead of polling the anchors via radio (which can be prone to interference), this script uses **seeded coordinates** to define the flight path. 

To ensure hardware longevity, the drone maintains a **safety buffer** from the physical anchors at all times.

### Mission Sequence
Upon execution, the drone will:
1. **Initialize Seeded Coordinates:** Load the $(X, Y, Z)$ positions of all LPS anchors from the local configuration file.
2. **Sequential Targeting:** Sort anchors by ID to visit them in ascending order (Node 0, Node 1, etc.).
3. **Safety-Buffered Navigation:** Calculate a path to each anchor but stop at a predefined distance (Safety Buffer) to prevent physical collisions with the infrastructure.
4. **Altitude Stability:** Maintain a constant `HOVER_HEIGHT` of 1.0m throughout the mission.
5. **Final Approach:** Navigate to the designated landing zone at `(0, 1)`.
6. **Precision Landing:** Execute a vertical descent to `(0, 1, 0)`.

---

##  How to Run

### 1. Prerequisites
Ensure you have the Crazyflie python library dependencies installed:
```bash
pip install cflib
