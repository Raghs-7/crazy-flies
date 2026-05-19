#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import math
import time


class CrazyflieRelativeCircle(Node):

    def __init__(self):
        super().__init__('crazyflie_circle')

        self.cmd_pub = self.create_publisher(Twist, '/crazyflie/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/crazyflie/odom', self.odom_callback, 10)

        # Current live position
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # Saved baseline configuration
        self.initial_x = None
        self.initial_y = None
        self.initial_z = None

        self.start_time = time.time()
        self.timer = self.create_timer(0.05, self.control_loop)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        # Capture initial spawn point on the very first frame
        if self.initial_x == None:
            self.initial_x = self.x
            self.initial_y = self.y
            self.initial_z = self.z
            self.get_logger().info(f"Origin Locked -> Initial X:{self.initial_x:.2f}, Y:{self.initial_y:.2f}")

        # Calculate relative coordinates (how far we have moved from spawn)
        rel_x = self.x - self.initial_x
        rel_y = self.y - self.initial_y
        rel_z = self.z - self.initial_z

        self.get_logger().info(
            f'ABS -> x:{self.x:.2f} y:{self.y:.2f} z:{self.z:.2f} | RELATIVE -> x:{rel_x:.2f} y:{rel_y:.2f} z:{rel_z:.2f}'
        )

    def control_loop(self):
        # Wait until we have safely intercepted odometry
        if self.initial_x is None:
            return

        msg = Twist()
        elapsed = time.time() - self.start_time

        # Calculate live relative offsets
        rel_x = self.x - self.initial_x
        rel_y = self.y - self.initial_y

        # -------------------------------------------------------------
        # PHASE 1 : VERTICAL TAKEOFF (No drift allowed)
        # -------------------------------------------------------------
        if elapsed < 5.0:
            target_rel_z = 0.6
            msg.linear.z = 2.0 * (target_rel_z - (self.z - self.initial_z))
            
            # Lock position to landing footprint
            msg.linear.x = -1.0 * rel_x
            msg.linear.y = -1.0 * rel_y
            msg.angular.z = 0.0

        # -------------------------------------------------------------
        # PHASE 2 : MOVE SAFELY INTO LOCAL POSITIVE SPACE
        # Shift deeper into local positive zone relative to where it took off
        # -------------------------------------------------------------
        elif elapsed < 10.0:
            target_rel_x = 0.5
            target_rel_y = 0.5

            msg.linear.x = 1.0 * (target_rel_x - rel_x)
            msg.linear.y = 1.0 * (target_rel_y - rel_y)
            
            target_rel_z = 0.6
            msg.linear.z = 2.0 * (target_rel_z - (self.z - self.initial_z))

        # -------------------------------------------------------------
        # PHASE 3 : RELATIVE BODY-FRAME CIRCLE
        # -------------------------------------------------------------
        elif elapsed < 25.0:
            radius = 0.35 
            omega = 0.7   

            # Drive circle using body frame coordinates
            msg.linear.x = radius * omega  
            msg.linear.y = 0.0
            msg.angular.z = omega          

            # Extreme altitude preservation
            target_rel_z = 0.6
            msg.linear.z = 2.5 * (target_rel_z - (self.z - self.initial_z))

        # -------------------------------------------------------------
        # PHASE 4 : LANDING
        # -------------------------------------------------------------
        elif elapsed < 30.0:
            target_rel_z = 0.02
            msg.linear.z = 1.0 * (target_rel_z - (self.z - self.initial_z))
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.0

        # -------------------------------------------------------------
        # TRAJECTORY COMPLETE
        # -------------------------------------------------------------
        else:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = -0.4
            self.cmd_pub.publish(msg)

            self.get_logger().info("Relative First Octant Flight Complete!")
            self.destroy_node()
            rclpy.shutdown()
            return

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieRelativeCircle()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
