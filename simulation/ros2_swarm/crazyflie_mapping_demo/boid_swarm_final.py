#!/usr/bin/env python3
"""
boid_swarm_final.py
────────────────────────────────────────────────────────────────────────────
Two Crazyflie drones flying as a boid swarm in Gazebo.

Phases:
  1. ENABLE   — sends gz enable signal to both drones
  2. TAKEOFF  — both climb to TARGET_Z using P-controller
  3. SWARM    — boid rules: separation + cohesion + goal attraction
  4. LAND     — both descend when close enough to GOAL

Run:
  source ~/crazyflie_mapping_demo/ros2_ws/install/setup.bash
  python3 ~/crazyflie_mapping_demo/boid_swarm_final.py
────────────────────────────────────────────────────────────────────────────
"""

import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ── Swarm settings ───────────────────────────────────────────────────────────

GOAL          = np.array([2.0, 2.0])   # XY goal both drones fly toward
TARGET_Z      = 0.8                    # cruise altitude (m)
LAND_RADIUS   = 0.25                   # land when both drones within this of GOAL

SAFE_DIST     = 0.7                    # minimum separation distance (m)
CONTROL_RATE  = 0.05                   # control loop period (s) = 20 Hz

# Boid weights
SEPARATION_W  = 0.8
COHESION_W    = 0.15
GOAL_W        = 0.4

MAX_SPEED_XY  = 0.25                   # m/s
MAX_SPEED_Z   = 0.5                    # m/s

# Takeoff: wait this many ticks at TARGET_Z before switching to swarm
TAKEOFF_TICKS = 60                     # 60 × 0.05 s = 3 s


# ── Boid force functions ─────────────────────────────────────────────────────

def separation(pos_self, pos_other):
    diff = pos_self - pos_other
    dist = np.linalg.norm(diff)
    if 1e-6 < dist < SAFE_DIST:
        return (diff / dist) * ((SAFE_DIST - dist) / SAFE_DIST)
    return np.zeros(2)

def cohesion(pos_self, pos_other):
    return (pos_other - pos_self) * 0.5

def goal_attraction(pos_self):
    vec  = GOAL - pos_self
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros(2)

def clamp(vec, limit):
    norm = np.linalg.norm(vec)
    return vec / norm * limit if norm > limit else vec


# ── Enable helper ────────────────────────────────────────────────────────────

def gz_enable(drone_name: str):
    subprocess.run([
        "gz", "topic",
        "-t", f"/{drone_name}/enable",
        "-m", "gz.msgs.Boolean",
        "-p", "data: true"
    ], check=False)


# ── Main swarm node ──────────────────────────────────────────────────────────

class BoidSwarm(Node):

    def __init__(self):
        super().__init__('boid_swarm')

        self.pos_cf1 = None
        self.pos_cf2 = None
        self.phase   = "enable"
        self.ticks   = 0

        # Subscribers
        self.create_subscription(Odometry, '/cf1/odom', self._odom_cf1, 10)
        self.create_subscription(Odometry, '/cf2/odom', self._odom_cf2, 10)

        # Publishers
        self.pub_cf1 = self.create_publisher(Twist, '/cf1/cmd_vel', 10)
        self.pub_cf2 = self.create_publisher(Twist, '/cf2/cmd_vel', 10)

        # Control loop
        self.create_timer(CONTROL_RATE, self._loop)
        self.get_logger().info("Boid swarm node started")

    # ── Odom callbacks ───────────────────────────────────────────────────────

    def _odom_cf1(self, msg):
        p = msg.pose.pose.position
        self.pos_cf1 = np.array([p.x, p.y, p.z])

    def _odom_cf2(self, msg):
        p = msg.pose.pose.position
        self.pos_cf2 = np.array([p.x, p.y, p.z])

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _stop(self):
        self.pub_cf1.publish(Twist())
        self.pub_cf2.publish(Twist())

    def _z_cmd(self, current_z):
        """P-controller for altitude hold."""
        return float(np.clip(1.2 * (TARGET_Z - current_z),
                             -MAX_SPEED_Z, MAX_SPEED_Z))

    # ── Main control loop ────────────────────────────────────────────────────

    def _loop(self):

        # ── ENABLE phase: send gz enable then wait for odom ──────────────────
        if self.phase == "enable":
            self.get_logger().info("Sending enable signals…")
            gz_enable("cf1")
            gz_enable("cf2")
            self.phase = "wait_odom"
            return

        if self.phase == "wait_odom":
            if self.pos_cf1 is not None and self.pos_cf2 is not None:
                self.get_logger().info("Odom received — starting takeoff")
                self.phase = "takeoff"
            return

        # Guard: need odom for all subsequent phases
        if self.pos_cf1 is None or self.pos_cf2 is None:
            return

        # ── TAKEOFF phase ────────────────────────────────────────────────────
        if self.phase == "takeoff":
            t1, t2 = Twist(), Twist()
            t1.linear.z = self._z_cmd(self.pos_cf1[2])
            t2.linear.z = self._z_cmd(self.pos_cf2[2])
            self.pub_cf1.publish(t1)
            self.pub_cf2.publish(t2)

            # Check both drones have reached TARGET_Z
            both_up = (abs(self.pos_cf1[2] - TARGET_Z) < 0.15 and
                       abs(self.pos_cf2[2] - TARGET_Z) < 0.15)
            self.ticks += 1
            if both_up and self.ticks > TAKEOFF_TICKS:
                self.get_logger().info("Takeoff complete — entering swarm mode")
                self.phase = "swarm"
            return

        # ── SWARM phase ──────────────────────────────────────────────────────
        if self.phase == "swarm":

            xy1 = self.pos_cf1[:2]
            xy2 = self.pos_cf2[:2]

            # CF1 boid forces
            vel1 = (SEPARATION_W * separation(xy1, xy2) +
                    COHESION_W   * cohesion(xy1, xy2)   +
                    GOAL_W       * goal_attraction(xy1))
            vel1 = clamp(vel1, MAX_SPEED_XY)

            # CF2 boid forces
            vel2 = (SEPARATION_W * separation(xy2, xy1) +
                    COHESION_W   * cohesion(xy2, xy1)   +
                    GOAL_W       * goal_attraction(xy2))
            vel2 = clamp(vel2, MAX_SPEED_XY)

            t1, t2 = Twist(), Twist()

            t1.linear.x = float(vel1[0])
            t1.linear.y = float(vel1[1])
            t1.linear.z = self._z_cmd(self.pos_cf1[2])

            t2.linear.x = float(vel2[0])
            t2.linear.y = float(vel2[1])
            t2.linear.z = self._z_cmd(self.pos_cf2[2])

            self.pub_cf1.publish(t1)
            self.pub_cf2.publish(t2)

            # Log distance to goal
            d1 = np.linalg.norm(xy1 - GOAL)
            d2 = np.linalg.norm(xy2 - GOAL)
            self.get_logger().info(
                f"CF1→goal: {d1:.2f}m  CF2→goal: {d2:.2f}m  "
                f"separation: {np.linalg.norm(xy1-xy2):.2f}m",
                throttle_duration_sec=1.0
            )

            # Switch to land when both are close to goal
            if d1 < LAND_RADIUS and d2 < LAND_RADIUS:
                self.get_logger().info("Goal reached — landing")
                self.phase = "land"
            return

        # ── LAND phase ───────────────────────────────────────────────────────
        if self.phase == "land":
            t1, t2 = Twist(), Twist()
            t1.linear.z = float(np.clip(-0.3, -MAX_SPEED_Z, MAX_SPEED_Z))
            t2.linear.z = float(np.clip(-0.3, -MAX_SPEED_Z, MAX_SPEED_Z))

            # Keep horizontal position stable during landing
            for drone, pos, pub in [
                ("cf1", self.pos_cf1, self.pub_cf1),
                ("cf2", self.pos_cf2, self.pub_cf2),
            ]:
                t = Twist()
                t.linear.z = -0.3
                pub.publish(t)

            if self.pos_cf1[2] < 0.08 and self.pos_cf2[2] < 0.08:
                self._stop()
                self.get_logger().info("Both drones landed. Mission complete!")
                self.phase = "done"
            return

        if self.phase == "done":
            self._stop()


# ── Entry point ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BoidSwarm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
