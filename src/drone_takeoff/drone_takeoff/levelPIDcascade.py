#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode, VehicleCommand, ActuatorMotors,
    VehicleLocalPosition, VehicleOdometry, VehicleThrustSetpoint
)
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Point, Vector3

import numpy as np
import time, math

# ---- constants / limits ----
G = 9.81
TILT_LIMIT_RAD = math.radians(45)
MIN_THRUST = 0.05
MAX_THRUST = 1.0
I_LEAK_TAU = 5.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

def quat_to_euler_zyx(q):
    w, x, y, z = q
    s = -2.0*(x*z - w*y)
    s = max(-1.0, min(1.0, s))
    pitch = math.asin(s)
    roll  = math.atan2( 2.0*(y*z + w*x), w*w - x*x - y*y + z*z )
    yaw   = math.atan2( 2.0*(x*y + w*z), w*w + x*x - y*y - z*z )
    return roll, pitch, yaw

def wrap_pi(a): return (a + math.pi) % (2.0*math.pi) - math.pi

class PIDcontrol(Node):
    def __init__(self):
        super().__init__('FullPIDControl_debug')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ---------- outer (pos->vel->acc) gains ----------
        self.USE_CASCADE = True
        # pos -> vel (XY/Z)
        self.Kp_pos_xy = 0.8;  self.Ki_pos_xy = 0.05; self.VEL_XY_MAX = 5.0
        self.Kp_pos_z  = 1.0;  self.Ki_pos_z  = 0.20; self.VEL_Z_MAX  = 2.0
        # vel -> acc (XY/Z)
        self.Kp_vel_xy = 1.6;  self.Ki_vel_xy = 0.10; self.A_XY_MAX   = 4.0
        self.Kp_vel_z  = 2.0;  self.Ki_vel_z  = 0.30; self.A_Z_MAX    = 6.0

        # old single-loop gains kept but XY I disabled (don’t fight the cascade)
        self.Kpx = 0.5; self.Kdx = 0.75; self.Kix = 0.0
        self.Kpy = 0.5; self.Kdy = 0.75; self.Kiy = 0.0
        self.Kpz = 0.65; self.Kdz = 0.7;  self.Kiz = 0.02  # Z only (unused if cascade on)

        # ---------- inner (att + rate) ----------
        self.Kp_eul   = np.array([0.25, 0.25, 0.20])
        self.Kd_body  = np.array([0.04, 0.03, 0.04])
        self.Ki_eul   = np.array([0.12, 0.12, 0.00])
        self.TORQUE_MAX = np.array([0.15, 0.15, 0.15])
        self.I_eul = np.array([0.0, 0.0, 0.0]); self.I_EUL_MAX = 0.3
        self.fcut_rates = 30.0
        self.p_lpf = self.q_lpf = self.r_lpf = 0.0

        # ---------- vehicle / mixer ----------
        self.m = 1.0
        self.u_hover = 0.73
        self.kf = (self.m*G)/(4*self.u_hover)
        self.km = 0.016
        self.propx = np.array([ +0.13, -0.13, +0.13, -0.13 ], float)
        self.propy = np.array([ +0.22, -0.20, -0.22, +0.20 ], float)
        self.s     = np.array([ +1,    +1,    -1,    -1    ], float)
        self.N = 4
        self.B = np.vstack([
            -self.propy * self.kf,
             self.propx * self.kf,
             self.s     * (self.km*self.kf),
            -np.ones(4) * self.kf
        ]).astype(float)
        l = np.sqrt((self.propx**2 + self.propy**2).mean()); kappa = 0.4
        self.Mx_max = self.My_max = (l/np.sqrt(2)) * (self.m*G) * kappa
        self.Mz_max = 0.15 * self.Mx_max
        self.T_max  = (self.m*G)/self.u_hover
        self.u_prev = np.zeros(4); self.slew_per_s = 40.0

        print("cond(B)=", np.linalg.cond(self.B))
        u0 = np.full(4, self.u_hover)
        print("w_hover ~ [0,0,0,-mG]:", np.round(self.B@u0, 3))

        # ---------- state ----------
        self.q = np.array([1.0, 0.0, 0.0, 0.0], float)
        self.omega = np.array([0.0,0.0,0.0], float)
        self.last_pos = np.zeros(3, float)
        self.vxa = self.vya = self.vza = 0.0

        self.error_x = self.error_y = self.error_z = 0.0
        self.yaw_d = math.radians(90.0)
        self.yaw_gate_rad = math.radians(10.0)

        # outer->inner handoff
        self.ax_cmd = 0.0; self.ay_cmd = 0.0; self.az_cmd = 0.0
        self.ref_vel = np.zeros(3, float)   # [vx,vy,vz]
        self.ref_acc = np.zeros(3, float)   # [ax,ay,az]
        self.sat_last = False

        # timing
        self.prev_time_outer = time.time(); self.dt_outer = 0.02
        self.prev_time_inner = time.time(); self.dt       = 0.0025

        # integrators (outer cascade)
        self.I_pos_xy = np.zeros(2, float); self.I_pos_z  = 0.0
        self.I_vel_xy = np.zeros(2, float); self.I_vel_z  = 0.0
        self.I_LIM_POS_XY = 2.0; self.I_LIM_POS_Z  = 2.0
        self.I_LIM_VEL_XY = 2.0; self.I_LIM_VEL_Z  = 2.0

        # ---------- trajectory (figure-8) ----------
        self.mode = 'fig8'
        self.center = np.array([1.0, 1.0, -5.0], float)
        self.Ax = 40.0; self.Ay = 40.0
        self.period = 40.0; self.w_traj = 2.0*math.pi/self.period
        self.phase = 0.0
        self.follow_tangent_yaw = True
        self.lock_center_on_first_pose = True
        self.have_center = False
        self.t0 = time.time()
        self.t_last = time.time()
        self.refpoint = [float(self.center[0]), float(self.center[1]), float(self.center[2])]

        # ---------- pubs / subs ----------
        self.ref_pub = self.create_publisher(Point,   '/debug/ref_xyz', 10)
        self.xyz_pub = self.create_publisher(Point,   '/debug/xyz',     10)
        self.rpy_pub = self.create_publisher(Vector3, '/debug/rpy',     10)
        self.rpy_ref_pub = self.create_publisher(Vector3, '/debug/rpy_ref', 10)

        self.u_pub      = self.create_publisher(Float32MultiArray, '/debug/u_motors', 10)
        self.u_mean_pub = self.create_publisher(Float32,           '/debug/u_mean',   10)
        self.sat_pub    = self.create_publisher(Float32,           '/debug/sat',      10)
        self.wcmd_pub   = self.create_publisher(Float32MultiArray, '/debug/w_cmd',    10)
        self.walloc_pub = self.create_publisher(Float32MultiArray, '/debug/w_alloc',  10)
        self.resid_pub  = self.create_publisher(Float32MultiArray, '/debug/residual', 10)

        self.cmd_pub   = self.create_publisher(VehicleCommand,        '/fmu/in/vehicle_command', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode', qos_profile)
        self.thrust_pub   = self.create_publisher(VehicleThrustSetpoint,'/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.act_motors_pub = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.cb_pos,  qos_profile)
        self.create_subscription(VehicleOdometry,      '/fmu/out/vehicle_odometry',          self.cb_odom, qos_profile)

        # ---------- timers ----------
        self.create_timer(0.05,  self.publish_offboard_mode)  # 20 Hz is plenty
        self.create_timer(0.02,  self.outer_step)             # 50 Hz outer
        self.create_timer(0.0025,self.inner_step)             # 400 Hz inner
        self.arm_timer = self.create_timer(2.0,  self.arm)
        self.offboard_tiemr = self.create_timer(3.0, self.set_offboard_mode)

    # ----- callbacks -----
    def cb_odom(self, msg: VehicleOdometry):
        self.q = np.array(msg.q, dtype=float)
        self.omega = np.array([msg.angular_velocity[0],
                               msg.angular_velocity[1],
                               msg.angular_velocity[2]], float)

    def cb_pos(self, msg: VehicleLocalPosition):
        self.last_pos[:] = [msg.x, msg.y, msg.z]
        self.vxa, self.vya, self.vza = msg.vx, msg.vy, msg.vz
        # debug at sensor rate is fine (lightweight)
        self.xyz_pub.publish(Point(x=msg.x, y=msg.y, z=msg.z))
        self.ref_pub.publish(Point(x=float(self.refpoint[0]),
                                   y=float(self.refpoint[1]),
                                   z=float(self.refpoint[2])))
        # current position error (used for quick print if needed)
        self.error_x = self.refpoint[0] - msg.x
        self.error_y = self.refpoint[1] - msg.y
        self.error_z = self.refpoint[2] - msg.z

    # ----- trajectory -----
    def update_reference(self):
        if self.mode != 'fig8': return
        now = time.time()
        dt  = now - self.t_last; self.t_last = now

        if self.lock_center_on_first_pose and not self.have_center and np.any(self.last_pos):
            self.center[0] = self.last_pos[0]; self.center[1] = self.last_pos[1]
            self.have_center = True

        t_cand = now - self.t0
        x_c = self.center[0] + self.Ax * math.sin(self.w_traj * t_cand)
        y_c = self.center[1] + self.Ay * math.sin(2.0 * self.w_traj * t_cand + self.phase)
        z_c = self.center[2]

        vx_ref =  self.Ax * self.w_traj * math.cos(self.w_traj * t_cand)
        vy_ref = 2.0 * self.Ay * self.w_traj * math.cos(2.0 * self.w_traj * t_cand + self.phase)
        ax_ref = -self.Ax * (self.w_traj**2) * math.sin(self.w_traj * t_cand)
        ay_ref = -(2.0*self.w_traj)**2 * self.Ay * math.sin(2.0 * self.w_traj * t_cand + self.phase)

        yaw_cand = math.atan2(vy_ref, vx_ref) if self.follow_tangent_yaw else 0.0

        _, _, yaw_act = quat_to_euler_zyx(self.q)
        if abs(wrap_pi(yaw_cand - yaw_act)) >= self.yaw_gate_rad:
            # freeze time
            self.t0 += dt
            # keep previous refpoint / ref_vel / ref_acc
            self.yaw_d = yaw_cand
            return

        # accept new reference
        self.refpoint = [x_c, y_c, z_c]
        self.ref_vel[:] = [vx_ref, vy_ref, 0.0]
        self.ref_acc[:] = [ax_ref, ay_ref, 0.0]
        self.yaw_d = yaw_cand

    # ----- outer 50 Hz -----
    def outer_step(self):
        now = time.time()
        self.dt_outer = max(1/100.0, min(now - self.prev_time_outer, 0.1))
        self.prev_time_outer = now

        if self.mode == 'fig8':
            self.update_reference()

        if self.USE_CASCADE:
            ax, ay, az = self.cascaded_outer_loops()
        else:
            # legacy PD+FF path (kept for completeness)
            ex = self.refpoint[0] - self.last_pos[0]
            ey = self.refpoint[1] - self.last_pos[1]
            ez = self.refpoint[2] - self.last_pos[2]
            dex_dt = self.ref_vel[0] - self.vxa
            dey_dt = self.ref_vel[1] - self.vya
            dez_dt = -self.vza
            ax_ff, ay_ff = self.ref_acc[0], self.ref_acc[1]
            ax = clamp(ax_ff + self.Kpx*ex + self.Kdx*dex_dt, -self.A_XY_MAX, self.A_XY_MAX)
            ay = clamp(ay_ff + self.Kpy*ey + self.Kdy*dey_dt, -self.A_XY_MAX, self.A_XY_MAX)
            az = clamp(self.Kpz*ez + self.Kdz*dez_dt,        -self.A_Z_MAX,  self.A_Z_MAX)

        # handoff to inner
        self.ax_cmd, self.ay_cmd, self.az_cmd = float(ax), float(ay), float(az)

    def cascaded_outer_loops(self):
        ex = self.refpoint[0] - self.last_pos[0]
        ey = self.refpoint[1] - self.last_pos[1]
        ez = self.refpoint[2] - self.last_pos[2]
        vx, vy, vz = float(self.vxa), float(self.vya), float(self.vza)
        dt = self.dt_outer
        leak = math.exp(-dt / max(1e-3, I_LEAK_TAU))

        # pos -> vel (with leak)
        self.I_pos_xy *= leak; self.I_pos_z *= leak
        self.I_pos_xy += np.array([ex, ey]) * dt
        self.I_pos_xy = np.clip(self.I_pos_xy, -self.I_LIM_POS_XY, self.I_LIM_POS_XY)
        self.I_pos_z  = float(np.clip(self.I_pos_z + ez*dt, -self.I_LIM_POS_Z, self.I_LIM_POS_Z))

        vx_cmd = self.ref_vel[0] + self.Kp_pos_xy*ex + self.Ki_pos_xy*self.I_pos_xy[0]
        vy_cmd = self.ref_vel[1] + self.Kp_pos_xy*ey + self.Ki_pos_xy*self.I_pos_xy[1]
        vz_cmd = self.ref_vel[2] + self.Kp_pos_z *ez + self.Ki_pos_z *self.I_pos_z

        v_xy = np.array([vx_cmd, vy_cmd], float)
        spd = np.linalg.norm(v_xy)
        if spd > self.VEL_XY_MAX: v_xy *= (self.VEL_XY_MAX/spd)
        vx_cmd, vy_cmd = v_xy.tolist()
        vz_cmd = float(np.clip(vz_cmd, -self.VEL_Z_MAX, self.VEL_Z_MAX))

        # vel -> acc (with leak)
        evx, evy, evz = vx_cmd - vx, vy_cmd - vy, vz_cmd - vz
        self.I_vel_xy *= leak; self.I_vel_z *= leak
        self.I_vel_xy += np.array([evx, evy]) * dt
        self.I_vel_xy = np.clip(self.I_vel_xy, -self.I_LIM_VEL_XY, self.I_LIM_VEL_XY)
        self.I_vel_z  = float(np.clip(self.I_vel_z + evz*dt, -self.I_LIM_VEL_Z, self.I_LIM_VEL_Z))

        ax = self.ref_acc[0] + self.Kp_vel_xy*evx + self.Ki_vel_xy*self.I_vel_xy[0]
        ay = self.ref_acc[1] + self.Kp_vel_xy*evy + self.Ki_vel_xy*self.I_vel_xy[1]
        az = self.ref_acc[2] + self.Kp_vel_z *evz + self.Ki_vel_z *self.I_vel_z

        ax = float(np.clip(ax, -self.A_XY_MAX, self.A_XY_MAX))
        ay = float(np.clip(ay, -self.A_XY_MAX, self.A_XY_MAX))
        az = float(np.clip(az, -self.A_Z_MAX,  self.A_Z_MAX))
        return ax, ay, az

    # ----- inner 400 Hz -----
    def inner_step(self):
        now = time.time()
        self.dt = max(1/800.0, min(now - self.prev_time_inner, 0.03))
        self.prev_time_inner = now

        # rate LPF
        alpha = math.exp(-2.0*math.pi*self.fcut_rates*self.dt)
        self.p_lpf = alpha*self.p_lpf + (1-alpha)*self.omega[0]
        self.q_lpf = alpha*self.q_lpf + (1-alpha)*self.omega[1]
        self.r_lpf = alpha*self.r_lpf + (1-alpha)*self.omega[2]

        # accel + yaw -> desired r,p and **thrust_norm**
        roll_d, pitch_d, thrust_norm = self.accel_yaw_to_rpy(self.ax_cmd, self.ay_cmd, self.az_cmd, self.yaw_d)
        u = clamp(thrust_norm, MIN_THRUST, MAX_THRUST)

        # attitude PID (+rate damping)
        tau = self.controller_PID_attitude(roll_d, pitch_d, self.yaw_d)

        # mix + publish
        u_mot, sat, w_cmd, w_alloc, resid = self.mix_to_motors(
            tau=np.array([tau[0], tau[1], tau[2]], float),
            u_thrust=u, dt=self.dt
        )
        self.sat_last = bool(sat)

        # debug pubs (light)
        self.rpy_ref_pub.publish(Vector3(x=float(roll_d), y=float(pitch_d), z=float(self.yaw_d)))
        roll, pitch, yaw = quat_to_euler_zyx(self.q)
        self.rpy_pub.publish(Vector3(x=float(roll), y=float(pitch), z=float(yaw)))

        arr = Float32MultiArray(); arr.data = u_mot.astype(np.float32).tolist(); self.u_pub.publish(arr)
        self.u_mean_pub.publish(Float32(data=float(np.mean(u_mot))))
        self.sat_pub.publish(Float32(data=1.0 if sat else 0.0))
        arr = Float32MultiArray(); arr.data = w_cmd.astype(np.float32).tolist();   self.wcmd_pub.publish(arr)
        arr = Float32MultiArray(); arr.data = w_alloc.astype(np.float32).tolist(); self.walloc_pub.publish(arr)
        arr = Float32MultiArray(); arr.data = resid.astype(np.float32).tolist();   self.resid_pub.publish(arr)

        now_us = int(self.get_clock().now().nanoseconds // 1000)
        th = VehicleThrustSetpoint(); th.timestamp = now_us; th.timestamp_sample = now_us
        th.xyz = [0.0, 0.0, -float(np.clip(np.mean(u_mot), 0.0, 1.0))]
        self.thrust_pub.publish(th)

        mot = ActuatorMotors()
        mot.timestamp = now_us; mot.timestamp_sample = now_us
        controls = list(map(float, u_mot)); controls += [float('nan')]*(12-len(controls))
        mot.control = controls; mot.reversible_flags = 0
        self.act_motors_pub.publish(mot)

        if sat:
            # bleed attitude I quickly
            self.I_eul *= (1.0 - self.dt/3.0)

    # ----- attitude ctrl & mapping & mixer -----
    def controller_PID_attitude(self, rolld, pitchd, yawd):
        roll, pitch, yaw = quat_to_euler_zyx(self.q)
        e = np.array([wrap_pi(rolld-roll), wrap_pi(pitchd-pitch), wrap_pi(yawd-yaw)], float)
        tau_p = (self.Kp_eul*e) - (self.Kd_body*np.array([self.p_lpf, self.q_lpf, self.r_lpf]))
        if not np.any(np.abs(tau_p) > self.TORQUE_MAX - 1e-6):
            self.I_eul = np.clip(self.I_eul + e*self.dt, -self.I_EUL_MAX, self.I_EUL_MAX)
        else:
            self.I_eul *= (1.0 - self.dt/3.0)
        tau = tau_p + self.Ki_eul*self.I_eul
        return np.clip(tau, -self.TORQUE_MAX, self.TORQUE_MAX)

    def accel_yaw_to_rpy(self, ax, ay, az, yaw_d):
        # NED specific force components (include gravity)
        fx, fy, fz = -ax, -ay, G - az
        pitch_d = math.atan2( fx*math.cos(yaw_d) + fy*math.sin(yaw_d),  fz )
        roll_d  = math.atan2( -fy*math.cos(yaw_d) + fx*math.sin(yaw_d), fz )
        roll_d  = clamp(roll_d,  -TILT_LIMIT_RAD, TILT_LIMIT_RAD)
        pitch_d = clamp(pitch_d, -TILT_LIMIT_RAD, TILT_LIMIT_RAD)
        thrust_norm = clamp(math.sqrt(fx*fx + fy*fy + fz*fz)/G, MIN_THRUST, MAX_THRUST)
        return roll_d, pitch_d, thrust_norm

    def mix_to_motors(self, tau, u_thrust, dt):
        Mx = np.clip(self.Mx_max*(tau[0]/0.15), -self.Mx_max, self.Mx_max)
        My = np.clip(self.My_max*(tau[1]/0.15), -self.My_max, self.My_max)
        Mz = np.clip(self.Mz_max*(tau[2]/0.15), -self.Mz_max, self.Mz_max)
        Fz = np.clip(-self.T_max*float(u_thrust), -self.T_max, 0.0)  # FRD up is negative
        w_des = np.array([Mx, My, Mz, Fz], float)

        u0  = np.full(self.N, self.u_hover, float)
        rhs = w_des - self.B@u0
        du  = np.linalg.lstsq(self.B, rhs, rcond=None)[0]
        u   = np.clip(u0 + du, 0.0, 1.0)

        if (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6):
            w2 = w_des.copy(); w2[2] *= 0.5
            rhs = w2 - self.B@u0
            u = np.clip(u0 + np.linalg.lstsq(self.B, rhs, rcond=None)[0], 0.0, 1.0)
            if (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6):
                w3 = w2.copy(); w3[3] = 0.7*w2[3] + 0.3*(-self.m*G)
                rhs = w3 - self.B@u0
                u = np.clip(u0 + np.linalg.lstsq(self.B, rhs, rcond=None)[0], 0.0, 1.0)

        du_max = self.slew_per_s * dt
        u = np.clip(self.u_prev + np.clip(u - self.u_prev, -du_max, du_max), 0.0, 1.0)
        self.u_prev = u

        sat = (u.min() <= 1e-6) or (u.max() >= 1.0-1e-6)
        w_alloc = self.B @ u
        residual = w_des - w_alloc
        return u, sat, w_des, w_alloc, residual

    # ----- PX4 mode / arm -----
    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = False; msg.velocity = False; msg.acceleration = False
        msg.attitude = False; msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = True
        self.offboard_pub.publish(msg)

    def arm(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0
        msg.target_system = 1; msg.target_component = 1
        msg.source_system = 1; msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Arm command sent")
        self.arm_timer.cancel()

    def set_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 6.0  # OFFBOARD
        msg.target_system = 1; msg.target_component = 1
        msg.source_system = 1; msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)
        self.get_logger().info("Set OFFBOARD mode command sent")
        self.offboard_tiemr.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = PIDcontrol()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
