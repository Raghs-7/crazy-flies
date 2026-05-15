# Crazyflie Autonomous Missions 🛸

This repository contains the Bitcraze Crazyflie python library integrated with custom autonomous flight sequences.

## 📍 Current Mission: Square Path with Spawn Detection
The primary script is designed to handle "random" spawn points. Upon execution, the drone will:
1. Identify its current LPS coordinates.
2. Output the spawn location to the terminal.
3. Navigate to the start point `(0, 1, 0.5)`.
4. Execute a 2x2 meter square flight path.
5. Return to the start and land.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have the Crazyflie python library dependencies installed:
```bash
pip install -e .
