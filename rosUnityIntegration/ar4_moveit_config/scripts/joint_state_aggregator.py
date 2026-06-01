#!/usr/bin/env python3
"""Joint state aggregator for dual AR4 setup.

Subscribes to /Robot1/joint_states and /Robot2/joint_states,
prefixes joint names (robot1_joint_1, robot2_joint_1, ...),
and republishes merged message to /joint_states at 50 Hz.

Required so the dual-robot robot_state_publisher and move_group_dual
see all 16 joints (8 per robot) in one topic with unique prefixed names.
"""

import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateAggregator(Node):
    def __init__(self):
        super().__init__("joint_state_aggregator")
        self._lock = threading.Lock()
        self._latest: dict[str, JointState] = {}

        for robot_id in ["Robot1", "Robot2"]:
            prefix = robot_id.lower() + "_"
            self.create_subscription(
                JointState,
                f"/{robot_id}/joint_states",
                lambda msg, p=prefix: self._callback(p, msg),
                10,
            )

        self._pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_timer(1.0 / 50.0, self._publish)

    def _callback(self, prefix: str, msg: JointState):
        prefixed = JointState()
        prefixed.header = msg.header
        prefixed.name = [prefix + n for n in msg.name]
        prefixed.position = list(msg.position)
        prefixed.velocity = list(msg.velocity)
        prefixed.effort = list(msg.effort)
        with self._lock:
            self._latest[prefix] = prefixed

    def _publish(self):
        with self._lock:
            msgs = list(self._latest.values())
        if not msgs:
            return
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "world"
        for m in msgs:
            out.name.extend(m.name)
            out.position.extend(m.position)
            out.velocity.extend(m.velocity)
            out.effort.extend(m.effort)
        self._pub.publish(out)


def main():
    rclpy.init()
    node = JointStateAggregator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
