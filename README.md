# Crazyflie Autonomous Missions 🛸

This repository contains the Bitcraze Crazyflie python library integrated with custom autonomous flight sequences, specifically designed for Loco Positioning System (LPS) environments.

## 📍 Current Mission: Sequential LPS Node Navigator
The primary script is designed for dynamic environment adaptation. Instead of following a hard-coded path, the drone queries the lab's infrastructure to define its flight plan. 

Upon execution, the drone will:
1. **Auto-Detect Anchors:** Retrieve the $(X, Y, Z)$ coordinates of all active LPS anchors directly from the drone's memory.
2. **Order of Operations:** Automatically sort the detected anchors by ID to visit them in ascending sequence (Node 0, Node 1, etc.).
3. **Safety-Buffered Flight:** Navigate to each anchor's $(X, Y)$ location while maintaining a constant `HOVER_HEIGHT` of 1.0m to avoid colliding with hardware.
4. **Final Approach:** Move to the designated landing zone at `(0, 1)`.
5. **Precision Landing:** Execute a vertical descent to `(0, 1, 0)`.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have the Crazyflie python library dependencies installed:
```bash
pip install cflib
