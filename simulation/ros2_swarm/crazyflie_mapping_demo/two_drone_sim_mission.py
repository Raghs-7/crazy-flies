#!/usr/bin/env python3
"""
two_drone_sim_mission.py
────────────────────────────────────────────────────────────────────────────
Autonomous checkpoint mission for TWO simulated Crazyflies in Gazebo.

Run:
  source ~/crazyflie_mapping_demo/ros2_ws/install/setup.bash
  python3 ~/crazyflie_mapping_demo/two_drone_sim_mission.py
────────────────────────────────────────────────────────────────────────────
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import subprocess
import threading
import time
import math


# ── Checkpoints ──────────────────────────────────────────────────────────────
CHECKPOINTS = [
    (0.0,  0.0, 0.5),
    (1.0,  0.0, 0.5),
    (1.0,  1.0, 0.5),
    (-1.0, 1.0, 0.5),
    (-1.0, 0.0, 0.8),
    (0.0,  0.0, 0.8),
    (0.0,  1.0, 0.5),
    (0.0,  0.0, 0.5),
]

CF2_ALTITUDE_OFFSET = 0.3
CF2_DELAY           = 10.0
ARRIVE_THRESHOLD    = 0.12
KP_XY               = 0.8
KP_Z                = 1.2
MAX_VEL_XY          = 0.4
MAX_VEL_Z           = 0.4
HOVER_PAUSE         = 2.0


def gz_enable(drone_name: str):
    """Send enable signal directly via gz CLI — same as doing it in terminal."""
    subprocess.run([
        "gz", "topic",
        "-t", f"/{drone_name}/enable",
        "-m", "gz.msgs.Boolean",
        "-p", "data: true"
    ], check=False)


class DroneController(Node):

    def __init__(self, name: str, cmd_topic: str, odom_topic: str,
                 altitude_offset: float = 0.0):
        super().__init__(name)
        self.drone_name      = name
        self.altitude_offset = altitude_offset

        self._pub = self.create_publisher(Twist, cmd_topic, 10)
        self._sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 10)

        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._got_odom = False

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._z = msg.pose.pose.position.z
        self._got_odom = True

    def wait_for_odom(self, timeout=10.0):
        t0 = time.time()
        while not self._got_odom:
            if time.time() - t0 > timeout:
                self.get_logger().error("Timed out waiting for odometry!")
                return False
            time.sleep(0.05)
        return True

    def _clamp(self, v, limit):
        return max(-limit, min(limit, v))

    def fly_to(self, tx, ty, tz):
        tz += self.altitude_offset
        rate = self.create_rate(20)
        self.get_logger().info(
            f"{self.drone_name} → ({tx:.2f}, {ty:.2f}, {tz:.2f})"
        )
        while rclpy.ok():
            ex = tx - self._x
            ey = ty - self._y
            ez = tz - self._z
            if math.sqrt(ex**2 + ey**2 + ez**2) < ARRIVE_THRESHOLD:
                break
            twist = Twist()
            twist.linear.x = self._clamp(KP_XY * ex, MAX_VEL_XY)
            twist.linear.y = self._clamp(KP_XY * ey, MAX_VEL_XY)
            twist.linear.z = self._clamp(KP_Z  * ez, MAX_VEL_Z)
            self._pub.publish(twist)
            rate.sleep()
        self._pub.publish(Twist())
        self.get_logger().info(f"{self.drone_name} reached checkpoint ✓")

    def hover(self, seconds: float):
        t0 = time.time()
        rate = self.create_rate(20)
        while time.time() - t0 < seconds and rclpy.ok():
            self._pub.publish(Twist())
            rate.sleep()

    def land(self):
        self.get_logger().info(f"{self.drone_name} landing…")
        rate = self.create_rate(20)
        while self._z > 0.08 and rclpy.ok():
            twist = Twist()
            twist.linear.z = -0.3
            self._pub.publish(twist)
            rate.sleep()
        self._pub.publish(Twist())
        self.get_logger().info(f"{self.drone_name} landed ✓")

    def run_mission(self, delay: float = 0.0):
        if not self.wait_for_odom():
            return

        if delay > 0:
            self.get_logger().info(
                f"{self.drone_name} waiting {delay}s before takeoff…"
            )
            time.sleep(delay)

        # Send enable signal — this is what was missing
        self.get_logger().info(f"{self.drone_name} sending enable signal…")
        gz_enable(self.drone_name)
        time.sleep(0.5)  # give Gazebo a moment to process enable

        for i, (x, y, z) in enumerate(CHECKPOINTS):
            self.get_logger().info(
                f"{self.drone_name} checkpoint {i+1}/{len(CHECKPOINTS)}"
            )
            self.fly_to(x, y, z)
            self.hover(HOVER_PAUSE)

        self.land()


def main():
    rclpy.init()

    cf1 = DroneController(
        name            = "cf1_mission",
        cmd_topic       = "/cf1/cmd_vel",
        odom_topic      = "/cf1/odom",
        altitude_offset = 0.0,
    )
    cf2 = DroneController(
        name            = "cf2_mission",
        cmd_topic       = "/cf2/cmd_vel",
        odom_topic      = "/cf2/odom",
        altitude_offset = CF2_ALTITUDE_OFFSET,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(cf1)
    executor.add_node(cf2)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    t1 = threading.Thread(target=cf1.run_mission, kwargs={"delay": 0.0})
    t2 = threading.Thread(target=cf2.run_mission, kwargs={"delay": CF2_DELAY})

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    cf1.get_logger().info("Both drones completed mission!")
    executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
