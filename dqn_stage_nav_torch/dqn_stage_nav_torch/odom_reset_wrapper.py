#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from std_srvs.srv import Empty


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q) -> float:
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def rot2d(x: float, y: float, yaw: float) -> Tuple[float, float]:
    """2D Rotation"""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


class OdomResetWrapper(Node):
    def __init__(self):
        super().__init__("odom_reset_wrapper")

        self.declare_parameter("sub_odom_topic", "/odom")
        self.declare_parameter("pub_odom_topic", "/odom/sim")
        self.declare_parameter("input_odom_topic", "/odom")
        self.declare_parameter("output_odom_topic", "/odom/sim")

        self.declare_parameter("stage_reset_service", "/reset_positions")
        self.declare_parameter("wrapper_reset_service", "/reset_sim")
        self.declare_parameter("zero_position_eps", 0.08)

        self.declare_parameter("spawn_global_x", -7.0)
        self.declare_parameter("spawn_global_y", -7.0)
        self.declare_parameter("spawn_global_yaw_deg", 45.0)
        self.declare_parameter("pub_raw_odom_topic", "") 

        # --- QoS  ---
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("qos_best_effort", False)
        self.declare_parameter("qos_transient_local", False)

        # --- For Diagnostic ---
        self.declare_parameter("pub_reset_count_topic", "") 

        sub_topic = str(self.get_parameter("sub_odom_topic").value)
        pub_topic = str(self.get_parameter("pub_odom_topic").value)

        # Fallback 
        if not sub_topic:
            sub_topic = str(self.get_parameter("input_odom_topic").value)
        if not pub_topic:
            pub_topic = str(self.get_parameter("output_odom_topic").value)

        self.input_odom_topic = sub_topic
        self.output_odom_topic = pub_topic

        self.stage_reset_service = str(self.get_parameter("stage_reset_service").value)
        self.wrapper_reset_service = str(self.get_parameter("wrapper_reset_service").value)
        self.zero_eps = float(self.get_parameter("zero_position_eps").value)

        self.spawn_x = float(self.get_parameter("spawn_global_x").value)
        self.spawn_y = float(self.get_parameter("spawn_global_y").value)
        self.spawn_yaw = math.radians(float(self.get_parameter("spawn_global_yaw_deg").value))
        self.pub_raw_odom_topic = str(self.get_parameter("pub_raw_odom_topic").value).strip()

        # QoS profile
        depth = int(self.get_parameter("qos_depth").value)
        best_effort = bool(self.get_parameter("qos_best_effort").value)
        transient_local = bool(self.get_parameter("qos_transient_local").value)

        reliability = ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE
        durability = DurabilityPolicy.TRANSIENT_LOCAL if transient_local else DurabilityPolicy.VOLATILE

        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, depth),
            reliability=reliability,
            durability=durability,
        )

        self.sub = self.create_subscription(Odometry, self.input_odom_topic, self._odom_cb, self.qos)
        self.pub = self.create_publisher(Odometry, self.output_odom_topic, self.qos)

        self.pub_raw = None
        if self.pub_raw_odom_topic:
            self.pub_raw = self.create_publisher(Odometry, self.pub_raw_odom_topic, self.qos)

        # Reset count publisher
        self.reset_count = 0
        self.pub_reset_count = None
        reset_topic = str(self.get_parameter("pub_reset_count_topic").value).strip()
        if reset_topic:
            self.pub_reset_count = self.create_publisher(Int32, reset_topic, self.qos)

        self.stage_reset_cli = self.create_client(Empty, self.stage_reset_service)
        self.wrapper_reset_srv = self.create_service(Empty, self.wrapper_reset_service, self._handle_reset)

        self._offset_x = 0.0
        self._offset_y = 0.0
        self._offset_yaw = 0.0
        self._need_rezero = True

        self.get_logger().info(f"[Wrapper] Sub: {self.input_odom_topic}  Pub: {self.output_odom_topic}")
        self.get_logger().info(f"[Wrapper] Stage reset srv: {self.stage_reset_service}")
        self.get_logger().info(f"[Wrapper] Wrapper reset srv: {self.wrapper_reset_service}")
        self.get_logger().info(
            f"[Wrapper] QoS: depth={self.qos.depth} reliability={'BEST_EFFORT' if best_effort else 'RELIABLE'}"
        )
        if self.pub_raw:
            self.get_logger().info(f"[Wrapper] Pub global (computed): {self.pub_raw_odom_topic}")
        if self.pub_reset_count:
            self.get_logger().info(f"[Wrapper] Pub reset_count: {reset_topic}")

    def _publish_reset_count(self) -> None:
        if self.pub_reset_count is None:
            return
        msg = Int32()
        msg.data = int(self.reset_count)
        self.pub_reset_count.publish(msg)

    def _handle_reset(self, request, response):
        if not self.stage_reset_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"[Wrapper] Stage reset service not available: {self.stage_reset_service}")
            return response

        self.stage_reset_cli.call_async(Empty.Request())
        self._need_rezero = True

        self.reset_count += 1
        self._publish_reset_count()
        return response

    def _capture_offsets_if_needed(self, msg: Odometry) -> None:
        if not self._need_rezero:
            return

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = float(yaw_from_quaternion(msg.pose.pose.orientation))

        if abs(x) < self.zero_eps and abs(y) < self.zero_eps and abs(wrap_pi(yaw)) < 0.10:
            self._offset_x = 0.0
            self._offset_y = 0.0
            self._offset_yaw = 0.0
        else:
            self._offset_x = x
            self._offset_y = y
            self._offset_yaw = yaw

        self._need_rezero = False

    def _publish_zeroed(self, msg: Odometry) -> Tuple[Odometry, float, float, float]:
        self._capture_offsets_if_needed(msg)

        x_in = float(msg.pose.pose.position.x)
        y_in = float(msg.pose.pose.position.y)
        yaw_in = float(yaw_from_quaternion(msg.pose.pose.orientation))

        x0 = x_in - self._offset_x
        y0 = y_in - self._offset_y
        yaw0 = wrap_pi(yaw_in - self._offset_yaw)

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.twist = msg.twist

        out.pose.pose.position.x = x0
        out.pose.pose.position.y = y0
        qx, qy, qz, qw = quaternion_from_yaw(yaw0)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        self.pub.publish(out)
        return out, x0, y0, yaw0

    def _publish_global_from_sim(self, sim_zeroed: Odometry, sx: float, sy: float, syaw: float) -> None:
        if self.pub_raw is None:
            return

        dx, dy = rot2d(sx, sy, self.spawn_yaw)
        gx = self.spawn_x + dx
        gy = self.spawn_y + dy
        gyaw = wrap_pi(self.spawn_yaw + syaw)

        out = Odometry()
        out.header = sim_zeroed.header
        out.child_frame_id = sim_zeroed.child_frame_id
        out.pose = sim_zeroed.pose
        out.twist = sim_zeroed.twist

        out.pose.pose.position.x = gx
        out.pose.pose.position.y = gy
        qx, qy, qz, qw = quaternion_from_yaw(gyaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        self.pub_raw.publish(out)

    def _odom_cb(self, msg: Odometry):
        sim_zeroed, sx, sy, syaw = self._publish_zeroed(msg)
        self._publish_global_from_sim(sim_zeroed, sx, sy, syaw)


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
            rclpy.shutdown()


if __name__ == "__main__":
    main()