#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleCommandAck, VehicleAttitudeSetpoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
import numpy as np
import time
import math

G = 9.81                          # m/s^2
TILT_LIMIT_RAD = math.radians(35) # safety tilt limit
MIN_THRUST = 0.05                 # normalized thrust limits for PX4
MAX_THRUST = 0.9

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

class PIDcontrol(Node):


    def __init__(self):
        super().__init__('PID_controller_cascade')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # match PX4 publisher
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_pub_in = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.armed = False
        self.in_offboard = False
        self.Kpz = 0.7
        self.Kiz = 0.1
        self.Kdz = 0.5
        self.Kpx = 0.2
        self.Kix = 0.1
        self.Kdx = 0.5
        self.Kpy = 0.1
        self.Kiy = 0.1
        self.Kdy = 0.5
        self.Vmax = 1.0
        self.error_x = 0.0
        self.error_y = 0.0
        self.error_z = 0.0
        self.have_pos = False
        self.setpoint_count = 0
        self.armed = False

    
        self.error_vx = 0.0
        self.error_vy = 0.0
        self.error_vz = 0.0
        self.yaw_d = 0.0
        
        self.vxPID = 0
        self.vyPID = 0
        self.vzPID = 0     
        self.vxa = 0
        self.vya = 0
        self.vza = 0   
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_pub_in)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_pub_in)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_pub_in)
        self.att_sp_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', qos_pub_in)

        self.prev_time = time.time()
        self.I = [0.0, 0.0, 0.0]
        self.Iv = [0.0, 0.0, 0.0]
        self.airborne = False

    


        self.reftraj = [
            [0.0,0.0,-5.0],   # decollo a 5m
            [5.0,0.0,-5.0],   # avanti 5m
            [5.0,5.0,-5.0],   # diagonale
            [0.0,0.0,-5.0],
            [0.0,0.0,0.0]    # ritorno
        ]

       
        self.refcount = 0
        self.refpoint = self.reftraj[self.refcount]
        
    
        # Timers
        self.create_timer(0.02, self.publish_offboard_mode)   # 10 Hz
        self.create_timer(0.02,  lambda: self.publish_setpoint())        # 10 Hz
        

        self.arm_timer = self.create_timer(2.0, self.arm)                     # arm after 2 s
        #self.offboard_tiemr = self.create_timer(4.0, self.set_offboard_mode)       # switch to offboard after 3 s
        self.mode_timer = self.create_timer(0.5, self.try_enter_offboard)

        self.current_position_recived = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',self.callback_pos,qos_profile_sensor_data)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.cb_status,
            qos_profile_sensor_data
        )

# (opzionale) ACK dei comandi, utile per vedere se OFFBOARD è accettato
        self.ack_sub = self.create_subscription(
            VehicleCommandAck,
            '/fmu/out/vehicle_command',
            self.cb_ack,
            qos_profile_sensor_data
        )

    def cb_status(self, msg: VehicleStatus):
        prev_armed = self.armed
        prev_offb = self.in_offboard
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_offboard = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        if self.armed != prev_armed or self.in_offboard != prev_offb:
            self.get_logger().info(f"[STATUS] armed={self.armed} offboard={self.in_offboard}")

    def cb_ack(self, msg: VehicleCommandAck):
        if msg.command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM:
            self.get_logger().info(f"ARM ACK result={msg.result}")  # 0 = ACCEPTED
        if msg.command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
            self.get_logger().info(f"SET_MODE ACK result={msg.result}")  # 0 = ACCEPTED




    def callback_pos(self,msg:VehicleLocalPosition):

        self.have_pos = True
        self.vxa = msg.vx
        self.vya = msg.vy
        self.vza = msg.vz 
        self.error_x = self.refpoint[0] - msg.x 
        self.error_y = self.refpoint[1] - msg.y
        self.error_z = self.refpoint[2] - msg.z
        

        print(f"l'errore in x è {self.error_x}")
        print(f"l'errore in y è {self.error_y}")
        print(f"l'errore in z è {self.error_z}")

       
        if msg.z < -0.2:
            self.airborne = True
        

        print(f"l'errore in x è {self.error_x}")
        print(f"l'errore in y è {self.error_y}")
        print(f"l'errore in z è {self.error_z}")
        # print(f"la distanza {distance}")
        if (abs(self.error_x) < 0.5 and abs(self.error_y) < 0.5 and abs(self.error_z) < 0.5):
            if not hasattr(self, "inside_since"):
                self.inside_since = time.time()   # just entered
            elif time.time() - self.inside_since > 1.0:  # stayed 1s
               self.refcount += 1
            if self.refcount < len(self.reftraj):
                    self.refpoint = self.reftraj[self.refcount]
                    print("POSIZIONE RAGGIUNTA")
        else:
    # reset timer if you go out of tolerance
            if hasattr(self, "inside_since"):
                del self.inside_since

            


    def controller_PID_position(self):

        now = time.time()
        self.dt = now - self.prev_time
        self.prev_time = now

        self.I[0] += self.error_x * self.dt
        self.I[1] += self.error_y * self.dt
        self.I[2] += self.error_z * self.dt

        dex_dt = -self.vxa
        dey_dt = -self.vya
        dez_dt = -self.vza

        self.axPID =  self.Kpx*self.error_x + self.Kdx*dex_dt + self.Kix*self.I[0] 
        self.ayPID =  self.Kpy*self.error_y + self.Kdy*dey_dt + self.Kiy*self.I[1]
        self.azPID =  self.Kpz*self.error_z + self.Kdz*dez_dt + self.Kiz*self.I[2]


    def accel_yaw_to_rpy(self, ax, ay, az, yaw_d):
        """
        Inputs in NED:
        ax, ay, az : desired translational accelerations (m/s^2), *without* gravity
        yaw_d      : desired yaw (rad)
        Returns:
        roll_d, pitch_d, thrust_norm
        """
        # include gravity, get total specific force in NED
        fx, fy, fz = ax, ay, az + G

        # Closed-form (ZYX: yaw-pitch-roll)
        pitch_d = math.atan2( fx*math.cos(yaw_d) + fy*math.sin(yaw_d),  fz )
        roll_d  = math.atan2( fy*math.cos(yaw_d) - fx*math.sin(yaw_d),  fz )

        # Tilt safety
        roll_d  = clamp(roll_d,  -TILT_LIMIT_RAD, TILT_LIMIT_RAD)
        pitch_d = clamp(pitch_d, -TILT_LIMIT_RAD, TILT_LIMIT_RAD)

        # Thrust (specific): ||f|| / g → normalized scalar
        T_over_m = math.sqrt(fx*fx + fy*fy + fz*fz)   # m/s^2
        thrust_norm = clamp(T_over_m / G, MIN_THRUST, MAX_THRUST)

        return roll_d, pitch_d, thrust_norm
    
    def rpy_to_quat(self, roll, pitch, yaw):
        cr = math.cos(roll*0.5);  sr = math.sin(roll*0.5)
        cp = math.cos(pitch*0.5); sp = math.sin(pitch*0.5)
        cy = math.cos(yaw*0.5);   sy = math.sin(yaw*0.5)

        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        return [w, x, y, z]

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds
        msg.position = False   # we want to control velocity
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        msg.thrust_and_torque = True

        self.offboard_pub.publish(msg)

    def publish_setpoint(self):

        self.controller_PID_position()
       
        ap= VehicleAttitudeSetpoint()
        ax, ay, az = self.axPID, self.ayPID, self.azPID

        # 2) accel + yaw -> roll, pitch, thrust
        roll_d, pitch_d, thrust_norm = self.accel_yaw_to_rpy(ax, ay, az, self.yaw_d)

        # 3) rpy -> quaternion (ZYX order)
        qd = self.rpy_to_quat(roll_d, pitch_d, self.yaw_d)

        # 4) fill PX4 attitude setpoint
        ap = VehicleAttitudeSetpoint()
        ap.timestamp = int(self.get_clock().now().nanoseconds//1000)
        ap.q_d = [qd[0], qd[1], qd[2], qd[3]]

        # thrust in FRD body frame: z is forward-right-down, so upward thrust is negative z
        ap.thrust_body[0] = 0.0
        ap.thrust_body[1] = 0.0
        takeoff_thrust = 0.90
        ap.thrust_body[2] = -takeoff_thrust
        self.setpoint_count += 1
        self.get_logger().info(
    f"armed={self.armed} offb={self.in_offboard} thrust_z={ap.thrust_body[2]:.2f}"
)

        """
        if not self.airborne:
            ap.thrust_body[2] = -takeoff_thrust   # FRD: negative lifts
        else:
            ap.thrust_body[2] = -clamp(thrust_norm, MIN_THRUST, MAX_THRUST)

        """

        # optional: keep yaw rate zero (we hold yaw by the quaternion already)
        ap.yaw_sp_move_rate = 0.0

        self.att_sp_pub.publish(ap)



    def arm(self):
        if self.armed:
            return
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
        self.arm_timer.cancel()  # STOP SEND THE LOG AFTER IT ARMS IT

    def try_enter_offboard(self):
        if self.in_offboard or not self.armed:
            return
        if not self.have_pos:
            return
        if self.setpoint_count < 25:  # ~0.5 s a 50 Hz
            return

        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds//1000)
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0   # base mode
        msg.param2 = 6.0   # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Richiesta OFFBOARD inviata (dopo stream setpoint)")



def main(args=None):
    rclpy.init(args=args)
    node = PIDcontrol()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
