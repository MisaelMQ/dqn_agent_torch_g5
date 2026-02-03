#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class OdomResetWrapper(Node):
    def __init__(self):
        super().__init__("odom_reset_wrapper")

        self.input_odom_topic = self.declare_parameter("input_odom_topic", "/odom").value
        self.output_odom_topic = self.declare_parameter("output_odom_topic", "/odom/sim").value

        self.stage_reset_service = self.declare_parameter("stage_reset_service", "/reset_positions").value
        self.wrapper_reset_service = self.declare_parameter("wrapper_reset_service", "/reset_sim").value

        self.sub = self.create_subscription(Odometry, self.input_odom_topic, self.odom_cb, 10)
        self.pub = self.create_publisher(Odometry, self.output_odom_topic, 10)

        self.stage_reset_cli = self.create_client(Empty, self.stage_reset_service)
        self.wrapper_reset_srv = self.create_service(Empty, self.wrapper_reset_service, self.handle_wrapper_reset)

        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0
        self.first_odom_after_reset = True

        self.get_logger().info(f"[Wrapper] Sub: {self.input_odom_topic}  Pub: {self.output_odom_topic}")
        self.get_logger().info(f"[Wrapper] Stage reset srv: {self.stage_reset_service}")
        self.get_logger().info(f"[Wrapper] Wrapper reset srv: {self.wrapper_reset_service}")

    def handle_wrapper_reset(self, request, response):
        if not self.stage_reset_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"[Wrapper] Stage reset service not available: {self.stage_reset_service}")
            return response

        self.stage_reset_cli.call_async(Empty.Request())
        self.get_logger().info("[Wrapper] Stage reset requested. Will re-zero odom on next odom message.")
        self.first_odom_after_reset = True
        return response

    def odom_cb(self, msg: Odometry):
        if self.first_odom_after_reset:
            self.offset_x = float(msg.pose.pose.position.x)
            self.offset_y = float(msg.pose.pose.position.y)
            self.offset_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
            self.first_odom_after_reset = False

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.twist = msg.twist

        out.pose.pose.position.x = float(msg.pose.pose.position.x) - self.offset_x
        out.pose.pose.position.y = float(msg.pose.pose.position.y) - self.offset_y

        yaw_in = yaw_from_quaternion(msg.pose.pose.orientation)
        yaw_out = wrap_pi(yaw_in - self.offset_yaw)
        qx, qy, qz, qw = quaternion_from_yaw(yaw_out)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomResetWrapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
