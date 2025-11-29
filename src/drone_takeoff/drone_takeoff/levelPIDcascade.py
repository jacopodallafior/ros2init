#!/usr/bin/env python3





# 2 LOOP PID CASCADE CONTROLLER FOR A DRONE WITH DIFFERENT PUBLISH TIME

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, ActuatorMotors,VehicleLocalPosition,VehicleAttitude,VehicleTorqueSetpoint,VehicleOdometry, VehicleThrustSetpoint, VehicleStatus,VehicleAttitudeSetpoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32,Float32MultiArray
from geometry_msgs.msg import Point, Vector3
#from nav_msgs.msg import Path
 

import numpy as np
import time
import math

G = 9.81                          # m/s^2
TILT_LIMIT_RAD = math.radians(45) # safety tilt limit
MIN_THRUST = 0.05                 # normalized thrust limits for PX4
MAX_THRUST = 1.0
HOVER_THRUST = 0.725   # da tarare; 0.5–0.6 in SITL è tipico
I_MAX = 3.0   
A_XY_MAX   = 50.0#5   # m/s^2 limit for horizontal accel command
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
        super().__init__('FullPIDControl_debug')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # match PX4 publisher
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Outer loop gains
        self.Kpz = 0.95#65
        self.Kiz = 0.02
        self.Kdz = 0.7#0.5
        
        self.Kpx = 0.5
        self.Kdx = 0.8
        self.Kix = 0.01
        self.Kpy = 0.5
        self.Kdy = 0.8
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
        self.Kp_eul   = np.array([0.25, 0.25, 0.2])   # disable yaw at first  0.25, 0.25, 0.2
        self.Kd_body  = np.array([0.03, 0.03, 0.04]) # rate damping # 0.10 0.10
        self.Ki_eul   = np.array([0.12, 0.12, 0.00])   # keep off for now 0.12, 0.12, 0.00
        self.TORQUE_MAX = np.array([0.15, 0.15, 0.15])  # NO yaw torque initially

        """
         self.Kp_eul   = np.array([0.20, 0.20, 0.2])   # disable yaw at first  0.25, 0.25, 0.2
        self.Kd_body  = np.array([0.01, 0.01, 0.04]) # rate damping # 0.10 0.10
        self.Ki_eul   = np.array([0.12, 0.12, 0.00])   # keep off for now 0.12, 0.12, 0.00
        self.TORQUE_MAX = np.array([0.15, 0.15, 0.15])  # NO yaw torque initially
        """
        self.I_eul = np.array([0.0, 0.0, 0.0])
        self.I_EUL_MAX = 0.3
        #self.TORQUE_MAX = np.array([1.0, 1.0, 1.0]) # normalized limits

        #FILTER ON D
        self.p_lpf = self.q_lpf = self.r_lpf = 0.0
        self.fcut_rates = 30.0 

        # MMA useful variables

        self.m = 1#0.6           # kg (put your mass)
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
            self.s  * (self.km*self.kf),   # Mz
            -np.ones(4) * self.kf     # Fz (FRD: up-thrust is negative)
        ]).astype(float)

        self.Brp = self.B[0:2, :]

        l = np.sqrt((self.propx**2 + self.propy**2).mean())         # approx arm (m)
        kappa = 0.4
        self.Mx_max = self.My_max = (l/np.sqrt(2)) * (self.m*G) * kappa
        self.Mz_max = 0.15 * self.Mx_max #0.3 before
        self.T_max  = (self.m*G)/self.u_hover
        # Slew limit helper
        self.u_prev = np.zeros(4)
        self.slew_per_s = 80.0  # max Δu per second (tune)

        print("cond(B)=", np.linalg.cond(self.B))       # deve essere basso (poche decine)
        u0  = np.full(4, self.u_hover)
        print("w_hover ~ [0,0,0,-mG]:", np.round(self.B@u0, 3))

        # Test yaw: con +Mz i CCW devono aumentare, i CW diminuire
        du_yaw = np.linalg.lstsq(self.B, np.array([0,0,0.2*self.Mz_max,-self.m*G]) - self.B@u0, rcond=None)[0]
        print("s*du_yaw (tutti >0 atteso):", np.round(self.s*du_yaw,3))

        # Test roll: con +Mx devono salire i motori con y<0 (2 e 3)
        du_roll = np.linalg.lstsq(self.B, np.array([0.2*self.Mx_max,0,0,-self.m*G]) - self.B@u0, rcond=None)[0]
        print("(-propy)*du_roll (tutti >0 atteso):", np.round((-self.propy)*du_roll,3))


        self.prev_time_outer = time.time()
        self.prev_time_inner = time.time()
        self.dt_outer = 0.01
        self.dt_inner = 1/400.0

        # outputs produced by the outer loop and consumed by the inner loop
        self.roll_d = 0.0
        self.pitch_d = 0.0
        self.u_thrust_cmd = HOVER_THRUST

        # DEBUG PUBLISHER
        self.dircos_pub   = self.create_publisher(Float32, '/debug/dir_cos', 10)
        # position / reference
        self.ref_pub      = self.create_publisher(Point, '/debug/ref_xyz', 10)
        self.xyz_pub      = self.create_publisher(Point, '/debug/xyz', 10)

        # attitude actual vs desired
        self.rpy_pub      = self.create_publisher(Vector3, '/debug/rpy', 10)
        self.rpy_ref_pub  = self.create_publisher(Vector3, '/debug/rpy_ref', 10)

        # motor usage + saturation + collective
        self.u_pub        = self.create_publisher(Float32MultiArray, '/debug/u_motors', 10)
        self.u_mean_pub   = self.create_publisher(Float32, '/debug/u_mean', 10)
        self.sat_pub      = self.create_publisher(Float32, '/debug/sat', 10)

        # allocator internals: commanded wrench, achieved wrench, residual
        self.wcmd_pub     = self.create_publisher(Float32MultiArray, '/debug/w_cmd', 10)   # [Mx, My, Mz, Fz]
        self.walloc_pub   = self.create_publisher(Float32MultiArray, '/debug/w_alloc', 10) # B @ u
        self.resid_pub    = self.create_publisher(Float32MultiArray, '/debug/residual', 10)


        
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
        self.yaw_gate_rad = math.radians(10.0)
        self.t_last = time.time()

    


                # ---- Trajectory generator (figure-8) ----
        # ---- Trajectory logic ----
        # start by going to (0,0,-5) as a waypoint, then switch to figure-8
        self.mode = 'waypoints'            # 'waypoints' or 'fig8'

        # single pre-trajectory waypoint: hover at (0,0,-5)
        self.reftraj = [
            [0.0, 0.0, -15.0],
        ]
        self.refcount = 0
        self.refpoint = list(self.reftraj[0])

        # figure-8 parameters (used when self.mode == 'fig8')
        self.center = np.array([0.0, 0.0, -15.0], dtype=float)  # center of the 8
        self.Ax = 40.0          # half-width in X (meters)
        self.Ay = 40.0          # half-height in Y (meters)

        self.period = 11   #40   # seconds (ω = 2π/period)
        self.w_traj = 2.0*math.pi / self.period
        self.phase = 0.0
        self.follow_tangent_yaw = True   # True → yaw points along motion, False → yaw=0
        self.lock_center_on_first_pose = True
        self.have_center = False

        self.t0 = time.time()


       
           
    
        # Timers
       
       # Timers  --- choose rates with time-scale separation
        self.create_timer(0.0025, self.publish_offboard_mode)  # 20 Hz
        self.create_timer(0.008, self.outer_loop)             # 100 Hz (position/trajectory)
        self.create_timer(0.005, self.inner_loop)            # 400 Hz (attitude/motors)

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
        '''
        # ---- Candidate position at t_cand ----
        x_cand = self.center[0] + self.Ax * math.sin(self.w_traj * t_cand)
        y_cand = self.center[1] + self.Ay * math.sin(2.0 * self.w_traj * t_cand + self.phase)
        z_cand = self.center[2]

        # REFERENCES FOR THE EIGTH TRAJECTORIES
        vx_ref =  self.Ax * self.w_traj * math.cos(self.w_traj * t_cand)
        vy_ref = 2.0 * self.Ay * self.w_traj * math.cos(2.0 * self.w_traj * t_cand + self.phase)
        ax_ref = -self.Ax * (self.w_traj**2) * math.sin(self.w_traj * t_cand)
        ay_ref = -(2.0*self.w_traj)**2 * self.Ay * math.sin(2.0 * self.w_traj * t_cand + self.phase)

        self.vx_ref, self.vy_ref = vx_ref, vy_ref
        self.ax_ref, self.ay_ref = ax_ref, ay_ref

        # ---- Candidate yaw at t_cand ----
        if self.follow_tangent_yaw:
            vx = self.Ax * self.w_traj * math.cos(self.w_traj * t_cand)
            vy = 2.0 * self.Ay * self.w_traj * math.cos(2.0 * self.w_traj * t_cand + self.phase)
            yaw_cand = math.atan2(vy, vx)
        else:
            yaw_cand = 0.0
        '''
         # ---- Candidate position at t_cand: CIRCLE ----
        theta = self.w_traj * t_cand

        x_cand = self.center[0] + self.Ax * math.cos(theta)
        y_cand = self.center[1] + self.Ay * math.sin(theta)
        z_cand = self.center[2]

        # REFERENCES FOR THE CIRCLE TRAJECTORY
        vx_ref = -self.Ax * self.w_traj * math.sin(theta)
        vy_ref =  self.Ay * self.w_traj * math.cos(theta)

        ax_ref = -self.Ax * (self.w_traj**2) * math.cos(theta)
        ay_ref = -self.Ay * (self.w_traj**2) * math.sin(theta)



        
        self.vx_ref, self.vy_ref = vx_ref, vy_ref
        self.ax_ref, self.ay_ref = ax_ref, ay_ref

        # ---- Candidate yaw at t_cand ----
        if self.follow_tangent_yaw:
           # vx = self.Ax * self.w_traj * math.cos(self.w_traj * t_cand)
           # vy = 2.0 * self.Ay * self.w_traj * math.cos(2.0 * self.w_traj * t_cand + self.phase)
            vx = -self.Ax * self.w_traj * math.sin(theta)
            vy =  self.Ay * self.w_traj * math.cos(theta)

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
        


    

    def callback_pos(self,msg:VehicleLocalPosition):
        self.vxa = msg.vx
        self.vya = msg.vy
        self.vza = msg.vz
        
        #if self.mode == 'fig8':
         #   self.update_reference()

        self.last_pos = np.array([msg.x, msg.y, msg.z], dtype=float)

        self.xyz_pub.publish(Point(x=msg.x, y=msg.y, z=msg.z))
        self.ref_pub.publish(Point(x=float(self.refpoint[0]),
                           y=float(self.refpoint[1]),
                           z=float(self.refpoint[2])))



        # Position errors w.r.t. current reference
        self.error_x = self.refpoint[0] - msg.x 
        self.error_y = self.refpoint[1] - msg.y
        self.error_z = self.refpoint[2] - msg.z


        # Only run the "advance to next waypoint" logic in waypoint mode
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
                    # reached current waypoint and stayed for 1 s
                    self.refcount += 1
                    del self.inside_since

                    if self.refcount < len(self.reftraj):
                        # move to next waypoint (if you add more later)
                        self.refpoint = self.reftraj[self.refcount]
                        print("POSIZIONE RAGGIUNTA (yaw ok)")
                    else:
                        # all waypoints done → switch to figure-8
                        print("All waypoints reached, switching to figure-8")
                        self.mode = 'fig8'

                        # fix the center of the 8 at the last waypoint (0,0,-5)
                        self.center[0] = self.refpoint[0]
                        self.center[1] = self.refpoint[1]
                        self.center[2] = self.refpoint[2]

                        # prevent update_reference from recentering on first pose
                        self.have_center = True
                        # restart figure-8 time
                        self.t0 = time.time()
                        self.t_last = time.time()
            else:
                if hasattr(self, "inside_since"):
                    del self.inside_since


            


    def controller_PID_position(self,dt):

                # derivative of error uses v_ref - v
        dex_dt = self.vx_ref - self.vxa if hasattr(self, 'vx_ref') else -self.vxa
        dey_dt = self.vy_ref - self.vya if hasattr(self, 'vy_ref') else -self.vya
        dez_dt = -self.vza

        ax_ff = self.ax_ref if hasattr(self, 'ax_ref') else 0.0
        ay_ff = self.ay_ref if hasattr(self, 'ay_ref') else 0.0

        self.axPID = ax_ff + self.Kpx*self.error_x + self.Kdx*dex_dt + self.Kix*self.I[0]
        self.ayPID = ay_ff + self.Kpy*self.error_y + self.Kdy*dey_dt + self.Kiy*self.I[1]
        self.azPD  = self.Kpz*self.error_z + self.Kdz*dez_dt

        self.axPIDclam = clamp(self.axPID, -A_XY_MAX, A_XY_MAX)
        self.ayPIDclam = clamp(self.ayPID, -A_XY_MAX, A_XY_MAX)

    def controller_PID_attitude(self,rolld,pitchd,yawd,dt):
        
        roll, pitch, yaw = quat_to_euler_zyx(self.q)
        e_roll  = wrap_pi(rolld  - roll)
        e_pitch = wrap_pi(pitchd - pitch)
        e_yaw   = wrap_pi(yawd   - yaw)
        e = np.array([e_roll, e_pitch, e_yaw], float)

        tau_p = (self.Kp_eul * e) - (self.Kd_body * np.array([self.p_lpf, self.q_lpf, self.r_lpf]))
        will_sat = np.any(np.abs(tau_p) > self.TORQUE_MAX - 1e-6)
        if not will_sat:
            self.I_eul = np.clip(self.I_eul + e*dt, -self.I_EUL_MAX, self.I_EUL_MAX)
        else:
            self.I_eul *= (1.0 - dt/3.0)

        tau = tau_p + self.Ki_eul*self.I_eul
        return np.clip(tau, -self.TORQUE_MAX, self.TORQUE_MAX)



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

    def outer_loop(self):
        # timing
        now = time.time()
        dt = now - self.prev_time_outer
        self.prev_time_outer = now
        dt = max(1/250.0, min(dt, 0.05))  # clip for robustness
        self.dt_outer = dt

        # update the moving reference (figure-8)
        if self.mode == 'fig8':
            self.update_reference()

        # position PID (+ feed-forward vel/acc)
        self.controller_PID_position(dt)

        # vertical command → collective thrust
        az_cmd = self.azPD + self.Kiz * self.I[2]
        u_unsat = HOVER_THRUST * (1.0 - az_cmd / G)
        u = clamp(u_unsat, MIN_THRUST, MAX_THRUST)

        # anti-windup on Z
        sat_hi = (u >= MAX_THRUST - 1e-6) and (az_cmd < 0)
        sat_lo = (u <= MIN_THRUST + 1e-6) and (az_cmd > 0)
        near_sp = (abs(self.error_z) < 0.05 and abs(self.vza) < 0.1)
        if not (sat_hi or sat_lo) and not near_sp:
            self.I[2] = clamp(self.I[2] + self.error_z*dt, -I_MAX, I_MAX)
        else:
            self.I[2] *= (1.0 - dt/5.0)

        # optional XY integral with clamping/leak
        if self.Kix > 0.0:
            near_xy = (abs(self.error_x) < 0.3 and abs(self.vxa) < 0.5)
            if abs(self.axPIDclam - self.axPID) < 1e-6 and near_xy:
                self.I[0] = clamp(self.I[0] + self.error_x*dt, -I_XY_MAX, I_XY_MAX)
            else:
                self.I[0] *= (1.0 - dt/I_LEAK_TAU)
        if self.Kiy > 0.0:
            near_yy = (abs(self.error_y) < 0.3 and abs(self.vya) < 0.5)
            if abs(self.ayPIDclam - self.ayPID) < 1e-6 and near_yy:
                self.I[1] = clamp(self.I[1] + self.error_y*dt, -I_XY_MAX, I_XY_MAX)
            else:
                self.I[1] *= (1.0 - dt/I_LEAK_TAU)

        # accel + yaw → desired roll,pitch and store for inner loop
        roll_d, pitch_d, _ = self.accel_yaw_to_rpy(self.axPIDclam, self.ayPIDclam, az_cmd, self.yaw_d)
        self.roll_d = roll_d
        self.pitch_d = pitch_d
        self.u_thrust_cmd = u  # collective used by inner loop

        # (optional) publish debug here at 100 Hz
        rollp, pitchp, yawp = quat_to_euler_zyx(self.q)
        self.rpy_pub.publish(Vector3(x=float(rollp),  y=float(pitchp),  z=float(yawp)))
        self.rpy_ref_pub.publish(Vector3(x=float(self.roll_d), y=float(self.pitch_d), z=float(self.yaw_d)))
        self.u_mean_pub.publish(Float32(data=float(self.u_thrust_cmd)))

    def inner_loop(self):
        # timing
        now = time.time()
        dt = now - self.prev_time_inner
        self.prev_time_inner = now
        dt = max(1/800.0, min(dt, 0.03))
        self.dt_inner = dt

        # rate LPF at inner-loop rate
        alpha = math.exp(-2.0*math.pi*self.fcut_rates*dt)
        self.p_lpf = alpha*self.p_lpf + (1-alpha)*self.omega[0]
        self.q_lpf = alpha*self.q_lpf + (1-alpha)*self.omega[1]
        self.r_lpf = alpha*self.r_lpf + (1-alpha)*self.omega[2]

        # attitude PID (+I) with body-rate damping
        tau = self.controller_PID_attitude(self.roll_d, self.pitch_d, self.yaw_d, dt)

        # mix & publish actuators every cycle
        u_mot, sat, w_cmd, w_alloc, resid = self.mix_to_motors(
            tau=np.array([tau[0], tau[1], tau[2]], float),
            u_thrust=self.u_thrust_cmd,
            dt=dt
        )

        # debug topics
        arr = Float32MultiArray(); arr.data = u_mot.astype(np.float32).tolist()
        self.u_pub.publish(arr)
        self.sat_pub.publish(Float32(data=1.0 if sat else 0.0))
        arr = Float32MultiArray(); arr.data = w_cmd.astype(np.float32).tolist();   self.wcmd_pub.publish(arr)
        arr = Float32MultiArray(); arr.data = w_alloc.astype(np.float32).tolist(); self.walloc_pub.publish(arr)
        arr = Float32MultiArray(); arr.data = resid.astype(np.float32).tolist();   self.resid_pub.publish(arr)

        # thrust setpoint (informational)
        now_us = int(self.get_clock().now().nanoseconds // 1000)
        th = VehicleThrustSetpoint()
        th.timestamp = now_us; th.timestamp_sample = now_us
        th.xyz = [0.0, 0.0, -float(np.clip(np.mean(u_mot), 0.0, 1.0))]
        self.thrust_pub.publish(th)

        # actuator_motors
        mot = ActuatorMotors()
        mot.timestamp = now_us; mot.timestamp_sample = now_us
        controls = list(map(float, u_mot)) + [float('nan')] * 8  # pad to 12
        mot.control = controls; mot.reversible_flags = 0
        self.act_motors_pub.publish(mot)

        if sat:
            # bleed integrators on saturation
            self.I_eul *= (1.0 - dt/3.0)     # inner
            self.I[0]  *= (1.0 - dt/5.0)     # outer XY
            self.I[1]  *= (1.0 - dt/5.0)
            self.I[2]  *= (1.0 - dt/5.0)


        
        self.prop1_debug_pub.publish(Float32(data=float(u_mot[0])))
        self.prop2_debug_pub.publish(Float32(data=float(u_mot[1])))
        self.prop3_debug_pub.publish(Float32(data=float(u_mot[2])))
        self.prop4_debug_pub.publish(Float32(data=float(u_mot[3])))

        # Debug actual thrust sent

    def publish_dir_cos(self, w_des, u_cmd):
        # desired roll–pitch torque from command
        t_des = w_des[0:2]              # [Mx_des, My_des]

        # allocated roll–pitch torque from actual motors
        t_alloc = self.Brp @ u_cmd      # 2x4 * 4 = R^2

        nd = np.linalg.norm(t_des)
        na = np.linalg.norm(t_alloc)

        cos_dir = 0.0
        if nd > 1e-4 and na > 1e-4:
            cos_dir = float(np.dot(t_des, t_alloc) / (nd * na))

        self.dircos_pub.publish(Float32(data=cos_dir))
        
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
        '''
        # Simple desaturation: drop yaw first, then relax thrust toward hover
        if (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6):
            w2 = w_des.copy(); w2[2] *= 0.5  # halve Mz
            rhs = w2 - self.B @ u0 # reccompute the 
            u = np.clip(u0+ np.linalg.lstsq(self.B, rhs, rcond=None)[0], 0.0, 1.0)
            if (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6):
                w3 = w2.copy()
                w3[3] = 0.9*w2[3] + 0.3*(-self.m*G)  # pull Fz toward hover  #0.7 
                rhs = w3 - self.B @ u0 # reccompute the 
                u = np.clip(u0+ np.linalg.lstsq(self.B, rhs, rcond=None)[0], 0.0, 1.0)
        '''
        # Slew limit
        du_max = self.slew_per_s * dt
        u = np.clip(self.u_prev + np.clip(u - self.u_prev, -du_max, du_max), 0.0, 1.0)
        self.u_prev = u
        self.publish_dir_cos(w_des, u)
        sat = (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6)
        w_alloc = self.B @ u
        residual = w_des - w_alloc
        return u, sat, w_des , w_alloc, residual


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
