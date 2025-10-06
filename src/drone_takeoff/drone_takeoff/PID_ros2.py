#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import time



class PIDcontrol(Node):


    def __init__(self):
        super().__init__('PID_controller')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # match PX4 publisher
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.Kpz = 0.9
        self.Kiz = 0.1
        self.Kdz = 0.3
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

        
        self.vxa = 0
        self.vya = 0
        self.vza = 0     
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

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
        self.create_timer(0.1, self.publish_offboard_mode)   # 10 Hz
        self.create_timer(0.1,  lambda: self.publish_setpoint())        # 10 Hz
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
        if abs(self.error_z) < 0.7 and abs(self.error_x) <0.7 and abs(self.error_y) < 0.7:

            self.refcount += 1
            if self.refcount < len(self.reftraj):
                self.refpoint = self.reftraj[self.refcount]
                print(f"POSIZIONE RAGGIUNTA")
            
            


    def controller_PID(self):

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        self.I[0] += self.error_x * dt
        self.I[1] += self.error_y * dt
        self.I[2] += self.error_z * dt

        dex_dt = -self.vxa
        dey_dt = -self.vya
        dez_dt = -self.vza

        self.vxa =  self.Kpx*self.error_x + self.Kdx*dex_dt + self.Kix*self.I[0] 
        self.vya =  self.Kpy*self.error_y + self.Kdy*dey_dt + self.Kiy*self.I[1]
        self.vza =  self.Kpz*self.error_z + self.Kdz*dez_dt + self.Kiz*self.I[2]



        print(f"la tua velocità in x è {self.vxa}")
        print(f"la tua velocità in y è {self.vya}")
        print(f"la tua velocità in z è {self.vza}")
        
 
    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = False   # we want to control velocity
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_setpoint(self):

        self.controller_PID()
        sp = TrajectorySetpoint()
     
        sp.velocity[0] = self.vxa
        sp.velocity[1] = self.vya
        sp.velocity[2] = self.vza
        #sp.yaw = yaw
 # z (NED, so -10 = climb up 10 m)
        sp.yaw = 0.0
        self.setpoint_pub.publish(sp)

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
