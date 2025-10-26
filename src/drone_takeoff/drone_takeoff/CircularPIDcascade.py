#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition,VehicleAttitude,VehicleTorqueSetpoint,VehicleOdometry, VehicleThrustSetpoint, VehicleStatus,VehicleAttitudeSetpoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32
import numpy as np
import time
import math

G = 9.81                          # m/s^2
TILT_LIMIT_RAD = math.radians(45) # safety tilt limit
MIN_THRUST = 0.05                 # normalized thrust limits for PX4
MAX_THRUST = 0.9
HOVER_THRUST = 0.75   # da tarare; 0.5–0.6 in SITL è tipico
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
        super().__init__('CircularPIDcascade')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # match PX4 publisher
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.Kpz = 0.6#1.6
        self.Kiz = 0.02
        self.Kdz = 0.7#0.5
        
        self.Kpx = 0.3
        self.Kdx = 0.75
        self.Kix = 0.01
        self.Kpy = 0.3
        self.Kdy = 0.75
        self.Kiy = 0.01

        self.Vmax = 1.0
        self.error_x = 0.0
        self.error_y = 0.0
        self.error_z = 0.0
        self.yaw_d = 0.0
        self.vxa = 0
        self.vya = 0
        self.vza = 0   
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)    # current attitude (w, x, y, z), world←body
        self.omega = np.array([0.0, 0.0, 0.0], dtype=float)     # [p, q, r] rad/s in FRD
        self.I_att = np.array([0.0, 0.0, 0.0], dtype=float)     # integral on attitude error (optional)

        self.Kp_eul   = np.array([0.75, 0.75, 0.35])   # disable yaw at first
        self.Kd_body  = np.array([0.10, 0.10, 0.06]) # rate damping
        self.Ki_eul   = np.array([0.12, 0.12, 0.00])   # keep off for now
        self.TORQUE_MAX = np.array([0.15, 0.15, 0.15])  # NO yaw torque initially
        self.I_eul = np.array([0.0, 0.0, 0.0])
        self.I_EUL_MAX = 0.3
        #self.TORQUE_MAX = np.array([1.0, 1.0, 1.0]) # normalized limits


        
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.att_sp_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', qos_profile)
        self.thrust_debug_pub = self.create_publisher(Float32, '/debug/thrust_z', 10)
        self.torquez_debug_pub = self.create_publisher(Float32, '/debug/torquez', 10)
        self.torquey_debug_pub = self.create_publisher(Float32, '/debug/torquey', 10)
        self.torquex_debug_pub = self.create_publisher(Float32, '/debug/torquex', 10)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)


        self.prev_time = time.time()
        self.I = [0.0, 0.0, 0.0]
        self.yaw_gate_rad = math.radians(10.0)
        self.t_last = time.time()

    


                # ---- Trajectory generator (figure-8) ----
        self.mode = 'fig8'            # 'waypoints' or 'fig8'
        self.center = np.array([0.0, 0.0, -5.0], dtype=float)  # [x0, y0, z0] (z<0 in NED)
        self.Ax = 40.0                # half-width in X (meters)
        self.Ay = 40.0                # half-height in Y (meters)

        self.period = 30.0            # seconds (ω = 2π/period). Increase if accel is too high.
        self.w_traj = 2.0*math.pi / self.period
        self.phase = 0.0              # phase for Y in the Lissajous form
        self.follow_tangent_yaw = True   # True → yaw points along motion, False → yaw=0
        self.lock_center_on_first_pose = True  # center on first pose automatically
        self.have_center = False

        self.t0 = time.time()

        self.refpoint = [float(self.center[0]), float(self.center[1]), float(self.center[2])]
       
        self.refcount = 0
       
        
    
        # Timers
        self.create_timer(0.008, self.publish_offboard_mode)   # 10 Hz
        self.create_timer(0.008,  lambda: self.publish_setpoint())        # 10 Hz
        self.arm_timer = self.create_timer(2.0, self.arm)                     # arm after 2 s
        self.offboard_tiemr = self.create_timer(3.0, self.set_offboard_mode)       # switch to offboard after 3 s

        self.current_position_recived = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',self.callback_pos,qos_profile)
        self.odom_sub = self.create_subscription(VehicleOdometry,'/fmu/out/vehicle_odometry',self.cb_odom,qos_profile)


    def update_reference(self):
        if self.mode != 'fig8':
            return

        now = time.time()
        dt = now - self.t_last
        self.t_last = now

        # Optionally lock the center when first pose arrives
        if self.lock_center_on_first_pose and not self.have_center and hasattr(self, 'last_pos'):
            self.center[0] = self.last_pos[0]
            self.center[1] = self.last_pos[1]
            self.have_center = True

        # ---- candidate time along the trajectory ----
        t_cand = now - self.t0

        # ---- Candidate position at t_cand ----
        x_cand = self.center[0] + self.Ax * math.sin(self.w_traj * t_cand)
        y_cand = self.center[1] + self.Ay * math.sin(2.0 * self.w_traj * t_cand + self.phase)
        z_cand = self.center[2]

        # ---- Candidate yaw at t_cand ----
        if self.follow_tangent_yaw:
            vx = self.Ax * self.w_traj * math.cos(self.w_traj * t_cand)
            vy = 2.0 * self.Ay * self.w_traj * math.cos(2.0 * self.w_traj * t_cand + self.phase)
            yaw_cand = math.atan2(vy, vx)
        else:
            yaw_cand = 0.0

        # ---- Check yaw gate vs actual yaw ----
        _, _, yaw_act = quat_to_euler_zyx(self.q)
        yaw_err = abs(wrap_pi(yaw_cand - yaw_act))

        if yaw_err >= self.yaw_gate_rad:
            # Freeze trajectory progress: move t0 forward by dt so (now - t0) doesn't change
            self.t0 += dt
            if not hasattr(self, 'refpoint'):
                # seed once so downstream code has a reference
                self.refpoint = [x_cand, y_cand, z_cand]
                self.yaw_d = yaw_cand
            return
            # keep previous reference; do not update refpoint/yaw_d
            

        # ---- Accept candidate as the new reference ----
        self.refpoint = [x_cand, y_cand, z_cand]
        self.yaw_d = yaw_cand



    def cb_odom(self, msg: VehicleOdometry):
        # PX4 ROS 2: msg.q is [w,x,y,z] (world←body)
        self.q = np.array(msg.q, dtype=float)

        # Body rates [p,q,r] in rad/s, body FRD
        # msg.angular_velocity is a float[3]
        self.omega = np.array([msg.angular_velocity[0],
                            msg.angular_velocity[1],
                            msg.angular_velocity[2]], dtype=float)
        


    

    def callback_pos(self, msg: VehicleLocalPosition):
        # Cache current velocity & pose
        self.vxa = msg.vx
        self.vya = msg.vy
        self.vza = msg.vz
        self.last_pos = (msg.x, msg.y, msg.z)

        # Keep the reference fresh for continuous trajectories
        if self.mode == 'fig8':
            self.update_reference()

        # Position errors w.r.t. current reference
        self.error_x = self.refpoint[0] - msg.x 
        self.error_y = self.refpoint[1] - msg.y
        self.error_z = self.refpoint[2] - msg.z

        print(f"l'errore in x è {self.error_x}")
        print(f"l'errore in y è {self.error_y}")
        print(f"l'errore in z è {self.error_z}")

        # Only run the "advance to next waypoint" logic in waypoint mode
        if self.mode == 'waypoints':
            # current yaw
            _, _, yaw = quat_to_euler_zyx(self.q)
            # desired yaw for waypoint mode (use whatever you want; here we use self.yaw_d)
            yaw_err = abs(wrap_pi(self.yaw_d - yaw))
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

    def controller_PID_attitude(self,rolld,pitchd,yawd):
        
        [roll,pitch,yaw] = quat_to_euler_zyx(self.q)
        
        e_roll  = wrap_pi(rolld  - roll)
        e_pitch = wrap_pi(pitchd - pitch)
        e_yaw   = wrap_pi(yawd   - yaw)

        e = np.array([e_roll, e_pitch, e_yaw], dtype=float)

        # 3) optional integral (with simple anti-windup)
        self.I_eul += e * self.dt
        self.I_eul = np.clip(self.I_eul, -self.I_EUL_MAX, self.I_EUL_MAX)

        # 4) PD(+I) to body torques (normalized). D on body rates is fine.
        tau = (self.Kp_eul * e) \
            - (self.Kd_body * self.omega)\
            + (self.Ki_eul * self.I_eul)

        # 5) saturate to PX4 normalized limits
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
        fx, fy, fz = -ax, -ay, G - az#ax, ay, az + G
        

        # Closed-form (ZYX: yaw-pitch-roll)
        pitch_d = math.atan2( fx*math.cos(yaw_d) + fy*math.sin(yaw_d),  fz )
        roll_d  = math.atan2( -fy*math.cos(yaw_d) + fx*math.sin(yaw_d),  fz )

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
        msg.thrust_and_torque = True
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_setpoint(self):
        if self.mode == 'fig8':
            self.update_reference()
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
        roll_d, pitch_d, _ = self.accel_yaw_to_rpy(ax, ay, az, self.yaw_d)

        # ---- Inner loop: Euler-error PD(+I) with body-rate damping ----
        tau = self.controller_PID_attitude(roll_d, pitch_d, self.yaw_d)  # normalized torques

        # ---- Publish torque & thrust setpoints (MANDATORY every cycle) ----
        now_us = int(self.get_clock().now().nanoseconds // 1000)

        tr = VehicleTorqueSetpoint()
        tr.timestamp = now_us
        tr.xyz = [float(tau[0]), float(tau[1]), float(tau[2])]
        if hasattr(tr, 'timestamp_sample'):
            tr.timestamp_sample = now_us
        self.torque_pub.publish(tr)

        th = VehicleThrustSetpoint()
        th.timestamp = now_us
        th.xyz = [0.0, 0.0, float(-u)]   # FRD: negative z = upward thrust
        if hasattr(th, 'timestamp_sample'):
            th.timestamp_sample = now_us
        self.thrust_pub.publish(th)

        # Debug actual thrust sent
        self.thrust_debug_pub.publish(Float32(data=float(th.xyz[2])))
        self.torquez_debug_pub.publish(Float32(data=float(tr.xyz[2])))
        self.torquey_debug_pub.publish(Float32(data=float(tr.xyz[1])))
        self.torquex_debug_pub.publish(Float32(data=float(tr.xyz[0])))

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
