#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
import math, time

class ArmAndTakeoff(Node):
    def __init__(self):
        super().__init__('arm_and_takeoff')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # pubs
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)

        # --- Figure-8 parameters (tweak these) ---
        self.cx, self.cy = 0.0, 0.0          # center (NED x/y)
        self.z_hold = -5.0                   # altitude (NED down)
        self.A = 25.0                        # half width  -> total ~50 m
        self.B = 25.0                        # half height -> total ~50 m
        self.T = 40.0                        # period (s) for one full loop
        self.phase = 0.0                     # phase shift for y(t); try math.pi/2 to rotate lobes
        self.omega = 2.0 * math.pi / self.T  # rad/s

        # time ref
        self.t0 = time.monotonic()

        # timers: publish at 20 Hz for smoother tracking
        self.create_timer(0.05, self.publish_offboard_mode)
        self.create_timer(0.05, self.publish_setpoint)
        self.arm_timer = self.create_timer(2.0, self.arm)
        self.offboard_timer = self.create_timer(3.0, self.set_offboard_mode)

    # Continuous figure-8 setpoint with yaw facing velocity
    def publish_setpoint(self):
        t = time.monotonic() - self.t0
        w = self.omega

        # Lissajous 8
        x = self.cx + self.A * math.sin(w * t)
        y = self.cy + self.B * math.sin(2.0 * w * t + self.phase)
        z = self.z_hold

        # Path tangent (for yaw)
        dx = self.A * w * math.cos(w * t)
        dy = 2.0 * self.B * w * math.cos(2.0 * w * t + self.phase)
        yaw = math.atan2(dy, dx)  # face along motion

        sp = TrajectorySetpoint()
        sp.position[0] = float(x)
        sp.position[1] = float(y)
        sp.position[2] = float(z)
        sp.yaw = float(yaw)
        self.setpoint_pub.publish(sp)

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def arm(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Arm command sent")
        self.arm_timer.cancel()

    def set_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 6.0  # OFFBOARD
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Set OFFBOARD mode command sent")
        self.offboard_timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = ArmAndTakeoff()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
