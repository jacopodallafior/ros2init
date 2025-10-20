#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus,VehicleAttitudeSetpoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32
import numpy as np
import time
import math

G = 9.81                          # m/s^2
TILT_LIMIT_RAD = math.radians(35) # safety tilt limit
MIN_THRUST = 0.05                 # normalized thrust limits for PX4
MAX_THRUST = 0.9
HOVER_THRUST = 0.75   # da tarare; 0.5–0.6 in SITL è tipico
I_MAX = 3.0   
A_XY_MAX   = 4.0   # m/s^2 limit for horizontal accel command
I_XY_MAX   = 2.0   # cap on XY integrators
I_LEAK_TAU = 5.0   # s, for gentle integral leakage

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

        self.Kpz = 0.6#1.6
        self.Kiz = 0.02
        self.Kdz = 0.7#0.5
        
        self.Kpx = 0.2
        self.Kdx = 0.6
        self.Kix = 0.0
        self.Kpy = 0.1
        self.Kdy = 0.6
        self.Kiy = 0.0

        self.Vmax = 1.0
        self.error_x = 0.0
        self.error_y = 0.0
        self.error_z = 0.0
        self.yaw_d = 0.0
        self.vxa = 0
        self.vya = 0
        self.vza = 0   

        
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.att_sp_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', qos_profile)
        self.thrust_debug_pub = self.create_publisher(Float32, '/debug/thrust_z', 10)

        self.prev_time = time.time()
        self.I = [0.0, 0.0, 0.0]

    


        self.reftraj = [
            [0.0,0.0,-5.0],   # decollo a 5m
            [50.0,0.0,-5.0],   # avanti 5m
            [50.0,50.0,-5.0],   # diagonale
            [0.0,0.0,-5.0],
            [0.0,0.0,0.0]    # ritorno
        ]

       
        self.refcount = 0
        self.refpoint = self.reftraj[self.refcount]
        
    
        # Timers
        self.create_timer(0.02, self.publish_offboard_mode)   # 10 Hz
        self.create_timer(0.02,  lambda: self.publish_setpoint())        # 10 Hz
        self.arm_timer = self.create_timer(2.0, self.arm)                     # arm after 2 s
        self.offboard_tiemr = self.create_timer(3.0, self.set_offboard_mode)       # switch to offboard after 3 s


        self.current_position_recived = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',self.callback_pos,qos_profile)



    def callback_pos(self,msg:VehicleLocalPosition):
        self.vxa = msg.vx
        self.vya = msg.vy
        self.vza = msg.vz
        self.error_x = self.refpoint[0] - msg.x 
        self.error_y = self.refpoint[1] - msg.y
        self.error_z = self.refpoint[2] - msg.z

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

      #  self.I[0] += self.error_x * self.dt
       # self.I[1] += self.error_y * self.dt
        #self.I[2] += self.error_z * self.dt

        dex_dt = -self.vxa
        dey_dt = -self.vya
        dez_dt = -self.vza

        self.axPID =  self.Kpx*self.error_x + self.Kdx*dex_dt + self.Kix*self.I[0] 
        self.ayPID =  self.Kpy*self.error_y + self.Kdy*dey_dt + self.Kiy*self.I[1]
        #self.azPID =  self.Kpz*self.error_z + self.Kdz*dez_dt + self.Kiz*self.I[2]
        self.azPD =  self.Kpz*self.error_z + self.Kdz*dez_dt 

        self.axPIDclam = clamp(self.axPID, -A_XY_MAX, A_XY_MAX)
        self.ayPIDclam = clamp(self.ayPID, -A_XY_MAX, A_XY_MAX)



        print(f"la tua acc in x è {self.axPID}")
        print(f"la tua acc in y è {self.ayPID}")
        print(f"la tua acc in z è {self.azPD}")


    def accel_yaw_to_rpy(self, ax, ay, az, yaw_d):
        """
        Inputs in NED:
        ax, ay, az : desired translational accelerations (m/s^2), *without* gravity
        yaw_d      : desired yaw (rad)
        Returns:
        roll_d, pitch_d, thrust_norm
        """
        # include gravity, get total specific force in NED
        fx, fy, fz = -ax, ay, G - az#ax, ay, az + G
        

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
        msg.position = False   # we want to control velocity
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_setpoint(self):

        self.controller_PID_position()
       
        ap= VehicleAttitudeSetpoint()



        az_cmd = self.azPD + self.Kiz * self.I[2]   # accel desiderata (+giù)

        # mappa con hover: az=0 -> hover
        u_unsat = HOVER_THRUST * (1.0 - az_cmd / G)
        u = clamp(u_unsat, MIN_THRUST, MAX_THRUST)

        # anti-windup (blocca I se saturato nella direzione che peggiora)
        sat_hi = (u >= MAX_THRUST - 1e-6) and (az_cmd < 0)  # chiedi ancora più su
        sat_lo = (u <= MIN_THRUST + 1e-6) and (az_cmd > 0)  # chiedi ancora più giù
        near_sp = (abs(self.error_z) < 0.05 and abs(self.vza) < 0.1)

        if not (sat_hi or sat_lo) and not near_sp:
            self.I[2] += self.error_z * self.dt
            self.I[2] = clamp(self.I[2], -I_MAX, I_MAX)
        else:
            # leakage: scarica lentamente l'integratore per evitare derive lente
            self.I[2] *= (1.0 - self.dt/5.0)

        near_xy = (abs(self.error_x) < 0.3 and abs(self.vxa) < 0.5)
        if self.Kix > 0.0:
            if abs(self.axPIDclam - self.axPID) < 1e-6 and near_xy:
                self.I[0] = clamp(self.I[0] + self.error_x*self.dt, -I_XY_MAX, I_XY_MAX)
            else:
                self.I[0] *= (1.0 - self.dt/I_LEAK_TAU)

        if self.Kiy > 0.0:
            near_yy = (abs(self.error_y) < 0.3 and abs(self.vya) < 0.5)
            if abs(self.ayPIDclam - self.ayPID) < 1e-6 and near_yy:
                self.I[1] = clamp(self.I[1] + self.error_y*self.dt, -I_XY_MAX, I_XY_MAX)
            else:
                self.I[1] *= (1.0 - self.dt/I_LEAK_TAU)


        #ax, ay, az = self.axPID, self.ayPID, self.azPID
        ax, ay, az = self.axPIDclam, self.ayPIDclam, az_cmd

        # 2) accel + yaw -> roll, pitch, thrust
        roll_d, pitch_d, thrust_norm = self.accel_yaw_to_rpy(ax, ay, az, self.yaw_d)

        # 3) rpy -> quaternion (ZYX order)
        qd = self.rpy_to_quat(roll_d, pitch_d, self.yaw_d)

        
        ap.timestamp = int(self.get_clock().now().nanoseconds//1000)
        ap.q_d = [qd[0], qd[1], qd[2], qd[3]] # [1.0,0.0,0.0,0.0]# 

        # thrust in FRD body frame: z is forward-right-down, so upward thrust is negative z
        ap.thrust_body[0] = 0.0
        ap.thrust_body[1] = 0.0
        #takeoff_thrust = 0.8
        ap.thrust_body[2] = -u
        #ap.thrust_body[2] = -clamp(thrust_norm, MIN_THRUST, MAX_THRUST)
        ap.yaw_sp_move_rate = 0.0
        self.thrust_debug_pub.publish(Float32(data=float(ap.thrust_body[2])))

        

        self.att_sp_pub.publish(ap)

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
        self.arm_timer.cancel()  # STOP SEND THE LOG AFTER IT ARMS IT

    def set_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0  # base mode
        msg.param2 = 6.0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Set OFFBOARD mode command sent")
        self.offboard_tiemr.cancel()  #STOP SENDING THE LOG AFTER IT TAKEOFF


def main(args=None):
    rclpy.init(args=args)
    node = PIDcontrol()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
