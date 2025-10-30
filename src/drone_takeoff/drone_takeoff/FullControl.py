#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, ActuatorMotors,VehicleLocalPosition,VehicleAttitude,VehicleTorqueSetpoint,VehicleOdometry, VehicleThrustSetpoint, VehicleStatus,VehicleAttitudeSetpoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32
import numpy as np
import time
import math

G = 9.81                          # m/s^2
TILT_LIMIT_RAD = math.radians(45) # safety tilt limit
MIN_THRUST = 0.05                 # normalized thrust limits for PX4
MAX_THRUST = 1.0
HOVER_THRUST = 0.725   # da tarare; 0.5–0.6 in SITL è tipico
I_MAX = 3.0   
A_XY_MAX   = 6.0   # m/s^2 limit for horizontal accel command
I_XY_MAX   = 2.0   # cap on XY integrators
I_LEAK_TAU = 5.0   # s, for gentle integral leakage

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def quat_to_euler_zyx(q):
        # q = [w, x, y, z], world←body (PX4)
        w, x, y, z = q
        # ZYX (yaw-pitch-roll): yaw ψ about Z, pitch θ about Y, roll φ about X
        # guard numerical drift
        s = -2.0*(x*z - w*y)
        s = max(-1.0, min(1.0, s))
        pitch = math.asin(s)  # θ

        roll  = math.atan2( 2.0*(y*z + w*x), w*w - x*x - y*y + z*z )  # φ
        yaw   = math.atan2( 2.0*(x*y + w*z), w*w + x*x - y*y - z*z )  # ψ
        return roll, pitch, yaw

def wrap_pi(a):
        # wrap to [-pi, pi]
        a = (a + math.pi) % (2.0*math.pi) - math.pi
        return a




class PIDcontrol(Node):


    def __init__(self):
        super().__init__('FullControl')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # match PX4 publisher
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Outer loop gains
        self.Kpz = 0.6#1.6
        self.Kiz = 0.02
        self.Kdz = 0.7#0.5
        
        self.Kpx = 0.3
        self.Kdx = 0.75
        self.Kix = 0.01
        self.Kpy = 0.3
        self.Kdy = 0.75
        self.Kiy = 0.01

        # State variables
        self.yaw_gate_rad = math.radians(10.0)
        self.yaw_now = 0.0
        self.Vmax = 1.0
        self.error_x = 0.0
        self.error_y = 0.0
        self.error_z = 0.0
        #self.yaw_d = 0.0
        self.yaw_d = math.radians(90.0)
        self.vxa = 0
        self.vya = 0
        self.vza = 0   
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)    # current attitude (w, x, y, z), world←body
        self.omega = np.array([0.0, 0.0, 0.0], dtype=float)     # [p, q, r] rad/s in FRD
        self.I_att = np.array([0.0, 0.0, 0.0], dtype=float)     # integral on attitude error (optional)


        # Inner loop gains
        self.Kp_eul   = np.array([0.75, 0.75, 0.35])   # disable yaw at first
        self.Kd_body  = np.array([0.10, 0.10, 0.04]) # rate damping
        self.Ki_eul   = np.array([0.12, 0.12, 0.00])   # keep off for now
        self.TORQUE_MAX = np.array([0.15, 0.15, 0.15])  # NO yaw torque initially
        self.I_eul = np.array([0.0, 0.0, 0.0])
        self.I_EUL_MAX = 0.3
        #self.TORQUE_MAX = np.array([1.0, 1.0, 1.0]) # normalized limits

        #FILTER ON D
        self.p_lpf = self.q_lpf = self.r_lpf = 0.0
        self.fcut_rates = 20.0 

        # MMA useful variables

        self.m = 0.6           # kg (put your mass)
        self.u_hover = 0.73    # hover command (0..1)
        self.kf       = (self.m*G)/(4*self.u_hover)  # [N] thrust per motor at command=1
        self.km       = 0.016         # [m] yaw torque ratio (start 0.015–0.025)

        self.propx = np.array([ +0.13, -0.13, +0.13, -0.13 ], float)
        self.propy = np.array([ +0.22, -0.20, -0.22, +0.20 ], float)
        self.s = np.array([ +1, +1, -1, -1 ], float)

        self.N = 4

        self.B = np.vstack([
            -self.propy * self.kf,             # Mx
            self.propx * self.kf,             # My
            self.s  * self.km,   # Mz
            -np.ones(4) * self.kf     # Fz (FRD: up-thrust is negative)
        ]).astype(float)

        l = np.sqrt((self.propx**2 + self.propy**2).mean())         # approx arm (m)
        kappa = 0.4
        self.Mx_max = self.My_max = (l/np.sqrt(2)) * (self.m*G) * kappa
        self.Mz_max = 0.15 * self.Mx_max #0.3 before
        self.T_max  = (self.m*G)/self.u_hover
        # Slew limit helper
        self.u_prev = np.zeros(4)
        self.slew_per_s = 20.0  # max Δu per second (tune)


        
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.att_sp_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', qos_profile)
        self.thrust_debug_pub = self.create_publisher(Float32, '/debug/thrust_z', 10)
        self.prop1_debug_pub = self.create_publisher(Float32, '/debug/prop1', 10)
        self.prop2_debug_pub = self.create_publisher(Float32, '/debug/prop2', 10)
        self.prop3_debug_pub = self.create_publisher(Float32, '/debug/prop3', 10)
        self.prop4_debug_pub = self.create_publisher(Float32, '/debug/prop4', 10)

        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.act_motors_pub = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)


        self.prev_time = time.time()
        self.I = [0.0, 0.0, 0.0]

    


        self.reftraj = [
            [0.0,0.0,-5.0],   # decollo a 5m
            [0.0,10.0,-5.0],   # avanti 5m
            [0.0,5.0,-5.0],   # diagonale
            [0.0,0.0,-5.0],
            [0.0,0.0,-5.0]    # ritorno
        ]

       
        self.refcount = 0
        self.refpoint = self.reftraj[self.refcount]
        
    
        # Timers
        self.create_timer(0.0025, self.publish_offboard_mode)   # 10 Hz
        self.create_timer(0.0025,  lambda: self.publish_setpoint())        # 10 Hz
        self.arm_timer = self.create_timer(2.0, self.arm)                     # arm after 2 s
        self.offboard_tiemr = self.create_timer(3.0, self.set_offboard_mode)       # switch to offboard after 3 s

        self.current_position_recived = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',self.callback_pos,qos_profile)
        self.odom_sub = self.create_subscription(VehicleOdometry,'/fmu/out/vehicle_odometry',self.cb_odom,qos_profile)


    def cb_odom(self, msg: VehicleOdometry):
        # PX4 ROS 2: msg.q is [w,x,y,z] (world←body)
        self.q = np.array(msg.q, dtype=float)

        # Body rates [p,q,r] in rad/s, body FRD
        # msg.angular_velocity is a float[3]
        self.omega = np.array([msg.angular_velocity[0],
                            msg.angular_velocity[1],
                            msg.angular_velocity[2]], dtype=float)
        


    

    def callback_pos(self,msg:VehicleLocalPosition):
        self.vxa = msg.vx
        self.vya = msg.vy
        self.vza = msg.vz
        #self.yaw_d = msg.heading
        self.error_x = self.refpoint[0] - msg.x 
        self.error_y = self.refpoint[1] - msg.y
        self.error_z = self.refpoint[2] - msg.z

        # print(f"la distanza {distance}")

            # current yaw
        _, _, self.yaw_now = quat_to_euler_zyx(self.q)
            # desired yaw for waypoint mode (use whatever you want; here we use self.yaw_d)
        yaw_err = abs(wrap_pi(self.yaw_d - self.yaw_now))
        yaw_ok  = yaw_err < self.yaw_gate_rad

        pos_ok = (abs(self.error_x) < 0.5 and
                    abs(self.error_y) < 0.5 and
                    abs(self.error_z) < 0.5)

        if pos_ok and yaw_ok:
            if not hasattr(self, "inside_since"):
                self.inside_since = time.time()
            elif time.time() - self.inside_since > 1.0:
                self.refcount += 1
            if self.refcount < len(self.reftraj):
                self.refpoint = self.reftraj[self.refcount]
                print("POSIZIONE RAGGIUNTA (yaw ok)")
        else:
            if hasattr(self, "inside_since"):
                del self.inside_since
        
        if self.refcount == 2.0:
            self.yaw_d = math.radians(40.0)
        elif self.refcount == 3.0:
            self.yaw_d = math.radians(80.0)

            


    def controller_PID_position(self):

        now = time.time()
        self.dt = now - self.prev_time
        self.dt = max(1/800.0, min(self.dt, 0.03))  # per un target 400 Hz

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

    def controller_PID_attitude(self,rolld,pitchd,yawd):
        
        [roll,pitch,yaw] = quat_to_euler_zyx(self.q)
        
        e_roll  = wrap_pi(rolld  - roll)
        e_pitch = wrap_pi(pitchd - pitch)
        e_yaw   = wrap_pi(yawd   - yaw)

        e = np.array([e_roll, e_pitch, e_yaw], dtype=float)

        tau_p = (self.Kp_eul * e) - (self.Kd_body * np.array([self.p_lpf,self.q_lpf,self.r_lpf]))
        will_sat = np.any(np.abs(tau_p) > self.TORQUE_MAX - 1e-6)
        if not will_sat:
            self.I_eul = np.clip(self.I_eul + e*self.dt, -self.I_EUL_MAX, self.I_EUL_MAX)
        else:
            self.I_eul *= (1.0 - self.dt/3.0)
        tau = tau_p + self.Ki_eul*self.I_eul
        tau = np.clip(tau, -self.TORQUE_MAX, self.TORQUE_MAX)
        return tau



    def accel_yaw_to_rpy(self, ax, ay, az, yaw_d):
        """
        Inputs in NED:
        ax, ay, az : desired translational accelerations (m/s^2), *without* gravity
        yaw_d      : desired yaw (rad)
        Returns:
        roll_d, pitch_d, thrust_norm
        """
        # include gravity, get total specific force in NED
        fx, fy, fz = -ax, -ay, G - az#-ax, ay, - az + G
        

        # Closed-form (ZYX: yaw-pitch-roll)
        pitch_d = math.atan2( fx*math.cos(yaw_d) + fy*math.sin(yaw_d),  fz )
        roll_d  = math.atan2( -fy*math.cos(yaw_d) + fx*math.sin(yaw_d),  fz )  #- davanti al cos + al sin

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
        msg.attitude = False
        msg.thrust_and_torque = False
        msg.direct_actuator = True
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_setpoint(self):
        # Outer loops (unchanged)
        self.controller_PID_position()

        # ---- Vertical thrust mapping (unchanged idea) ----
        az_cmd = self.azPD + self.Kiz * self.I[2]
        u_unsat = HOVER_THRUST * (1.0 - az_cmd / G)
        u = clamp(u_unsat, MIN_THRUST, MAX_THRUST)

        # anti-windup on Z (unchanged)
        sat_hi = (u >= MAX_THRUST - 1e-6) and (az_cmd < 0)
        sat_lo = (u <= MIN_THRUST + 1e-6) and (az_cmd > 0)
        near_sp = (abs(self.error_z) < 0.05 and abs(self.vza) < 0.1)
        if not (sat_hi or sat_lo) and not near_sp:
            self.I[2] += self.error_z * self.dt
            self.I[2] = clamp(self.I[2], -I_MAX, I_MAX)
        else:
            self.I[2] *= (1.0 - self.dt/5.0)
        alpha = math.exp(-2.0*math.pi*self.fcut_rates*self.dt)
        self.p_lpf = alpha*self.p_lpf + (1-alpha)*self.omega[0]
        self.q_lpf = alpha*self.q_lpf + (1-alpha)*self.omega[1]
        self.r_lpf = alpha*self.r_lpf + (1-alpha)*self.omega[2]
        # optional XY integrators (unchanged)
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

        # ---- Accel + yaw → desired roll,pitch (for the inner loop) ----
        ax, ay, az = self.axPIDclam, self.ayPIDclam, az_cmd

        #self.yaw_d = self.yaw_now
        roll_d, pitch_d, _ = self.accel_yaw_to_rpy(ax, ay, az, self.yaw_d)  # not use yaw_now
        
        # ---- Inner loop: Euler-error PD(+I) with body-rate damping ----
        tau = self.controller_PID_attitude(roll_d, pitch_d, self.yaw_d)  # normalized torques

        # ---- Publish torque & thrust setpoints (MANDATORY every cycle) ----
        now_us = int(self.get_clock().now().nanoseconds // 1000)

        

        u_mot, sat = self.mix_to_motors(tau=np.array([tau[0], tau[1], tau[2]], float),
            u_thrust=u,
            dt=self.dt)

        if sat:
            # bleed attitude I quickly
            self.I_eul *= (1.0 - self.dt/3.0)
            # bleed outer-loop I gently
            self.I[0] *= (1.0 - self.dt/5.0)
            self.I[1] *= (1.0 - self.dt/5.0)
            self.I[2] *= (1.0 - self.dt/5.0)

        now_us = int(self.get_clock().now().nanoseconds // 1000)
        mot = ActuatorMotors()
        mot.timestamp = now_us
        mot.timestamp_sample = now_us
        #u_mot = [1,1,1,1]
        controls = list(map(float, u_mot))
        controls += [float('nan')] * (12 - len(controls))  # pad to 12
        mot.control = controls
        mot.reversible_flags = 0  # set bits if you truly use reversible ESCs
        self.act_motors_pub.publish(mot)
        
        self.prop1_debug_pub.publish(Float32(data=float(u_mot[0])))
        self.prop2_debug_pub.publish(Float32(data=float(u_mot[1])))
        self.prop3_debug_pub.publish(Float32(data=float(u_mot[2])))
        self.prop4_debug_pub.publish(Float32(data=float(u_mot[3])))

        # Debug actual thrust sent
        
        # NEW
    def mix_to_motors(self, tau, u_thrust, dt):
        """
        tau: np.array([tx,ty,tz]) from your controller (|tau| ≤ 0.15)
        u_thrust: scalar in [0,1] (your vertical command)
        dt: seconds
        returns: u[0..N-1] ∈ [0,1]
        """
        # Map your unitless tau to a physical wrench (N·m), clamp to limits
        Mx = np.clip(self.Mx_max * (tau[0] / 0.15), -self.Mx_max, self.Mx_max)
        My = np.clip(self.My_max * (tau[1] / 0.15), -self.My_max, self.My_max)
        Mz = np.clip(self.Mz_max * (tau[2] / 0.15), -self.Mz_max, self.Mz_max)
        Fz = np.clip(-self.T_max * float(u_thrust), -self.T_max, 0.0)  # up-thrust = negative

        w_des = np.array([Mx, My, Mz, Fz], float)

        # Least-squares allocation
        u0  = np.full(self.N, self.u_hover, float)
        rhs = w_des - self.B @ u0
        du  = np.linalg.lstsq(self.B, rhs, rcond=None)[0]
        u   = np.clip(u0 + du, 0.0, 1.0)
        print(f"la soluzione in ingresso è {du}")

        # Simple desaturation: drop yaw first, then relax thrust toward hover
        if (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6):
            w2 = w_des.copy(); w2[2] *= 0.5  # halve Mz
            u = np.clip(np.linalg.lstsq(self.B, w2, rcond=None)[0], 0.0, 1.0)
            if (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6):
                w3 = w2.copy()
                w3[3] = 0.7*w2[3] + 0.3*(-self.m*G)  # pull Fz toward hover
                u = np.clip(np.linalg.lstsq(self.B, w3, rcond=None)[0], 0.0, 1.0)

        # Slew limit
        du_max = self.slew_per_s * dt
        u = np.clip(self.u_prev + np.clip(u - self.u_prev, -du_max, du_max), 0.0, 1.0)
        self.u_prev = u

        sat = (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6)
        return u, sat


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
