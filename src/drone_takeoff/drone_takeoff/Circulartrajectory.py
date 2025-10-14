#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan  # TO AVOID PUT THE WHOLE TYPE Type: sensor_msgs/msg/LaserScan
import math 
import time
import numpy as np



class Trajectoryplanning(Node):


    def __init__(self):
        super().__init__('Circulartrajectory')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # match PX4 publisher
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.num = 1
        self.global_x = 0.0
        self.global_y = 0.0
        self.global_z = 0.0
        self.global_yaw = 0.0
        self.error_z = 0.0
        self.error_y = 0.0
        self.error_x = 0.0
        self.desired_yaw =  0.0
        self.yaw_target = 0.0
        
        self.n_points = 36
        self.wp_tol = 0.25          # [m] reach tolerance
        self.wp_hold_needed = 5     # scans in-a-row inside tol before advancing
        self.wp_hold_counter = 0
        self.orbit_idx = 0
        self.orbit_wps = None       # list of NED waypoints: [{"x","y","z","yaw"}, ...]


        self.d_safe   = 1.35   # [m] distanza di rispetto (0.30–0.40)
        self.delta_ok = 0.03   # [m] margine di fattibilità

     
        # Publishers
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)


        self.reftraj = [{"x": 0.0, "y": 0.0, "z": -5.0}
          
        ]
        self.refcount = 0



        self.current_position = {"x": 0.0, "y": 0.0, "z": -5.0}

       
        #self.current_position_wp = {"x": 0.0, "y": 0.0, "z": -5.0}

        # Timers
        self.create_timer(0.1, self.publish_offboard_mode)   # 10 Hz
        self.create_timer(0.1,  lambda: self.publish_setpoint(coordinate=self.current_position))        # 10 Hz
        self.arm_timer = self.create_timer(2.0, self.arm)                     # arm after 2 s
        # REMOVE the fixed offboard timer:  switch after N setpoints from inside publish_setpoint()
        self.offboard_tiemr = self.create_timer(3.0, self.set_offboard_mode)   #switch to offboard after 3 s


        self.current_position_recived = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',self.callback_pos,qos_profile)
        self.subscription = self.create_subscription(LaserScan,'/world/my_world/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan',self.lidar_callback,qos_profile)
         
  

    def callback_pos(self,msg:VehicleLocalPosition):
        
        #self.pose_ready = True
        
        #self.current_position= self.reftraj[self.refcount]
        

        self.global_x = msg.x 
        self.global_y = msg.y
        self.global_z = msg.z
        self.global_yaw = msg.heading
        print(f"l'altezza in z è {self.global_z}")
       # print(f"la posizione è {self.current_position}")
        

        self.error_x = self.current_position["x"] - msg.x 
        self.error_y = self.current_position["y"] - msg.y
        self.error_z = self.current_position["z"] - msg.z

        print(f"la posizione in x è {msg.x}")
        print(f"la posizione in y è {msg.y}")
        print(f"la posizione in z è {msg.z}")
        
       
    def _wrap_pi(self, a):
    
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def lidar_callback(self, msg: LaserScan):
        # ---------- 0) altitude first ----------
        if not hasattr(self, "phase"):
            self.phase = "TAKEOFF"
        target_altitude = -5.0  # NED (negative up)

        if self.phase == "TAKEOFF":
            altitude_error = abs(self.global_z - target_altitude)
            if altitude_error > 0.3:
                # only move vertically until at altitude
                self.current_position = {"x": self.global_x, "y": self.global_y, "z": target_altitude}
                self.get_logger().info(f"Climbing to altitude {target_altitude:.1f} (current {self.global_z:.2f})")
            else:
                self.phase = "APPROACH"
                self.get_logger().info("Altitude reached → switching to APPROACH phase")
            return

        # ---------- 1) LiDAR → body (take only the nearest contiguous cluster) ----------
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        if not np.any(valid):
            self.get_logger().info("No valid LiDAR returns.")
            return

        ranges_v = ranges[valid]
        angles_v = angles[valid]

            # ---------- 2) Find nearest point and cluster around it ----------
        i0 = int(np.argmin(ranges_v))
        r0 = ranges_v[i0]

        dr_thresh = 0.25  # [m] threshold for contiguous cluster
        left = i0
        while left - 1 >= 0 and abs(ranges_v[left - 1] - r0) < dr_thresh:
            left -= 1
        right = i0
        N = len(ranges_v)
        while right + 1 < N and abs(ranges_v[right + 1] - r0) < dr_thresh:
            right += 1

        r_cluster = ranges_v[left:right + 1]
        a_cluster = angles_v[left:right + 1]

        if r_cluster.size < 6:
            self.get_logger().info("Too few points in cluster.")
            return

        # ---------- 4) Fit circle with algebraic least squares ----------
            # Model: x^2 + y^2 + A*x + B*y + C = 0
            # body-aligned LiDAR (SDF has zero rotation), 2D Cartesian
        xs = r_cluster * np.cos(a_cluster)
        ys =-( r_cluster * np.sin(a_cluster) )# CONVENTION  FRD = [ x_FLU, -y_FLU, -z_FLU ]

        X = np.column_stack([xs, ys, np.ones_like(xs)])
        Z = -(xs ** 2 + ys ** 2)

        try:
            sol, *_ = np.linalg.lstsq(X, Z, rcond=None)
            A, B, C = sol
        except np.linalg.LinAlgError:
            self.get_logger().warn("Circle fit failed (singular matrix).")
            return

        a = -A / 2
        b = -B / 2
        R = math.sqrt(a ** 2 + b ** 2 - C)

        # ---------- 5) Diagnostics ----------
        self.get_logger().info(
            f"Circle fit: center=({a:.2f}, {b:.2f}), R={R:.2f}, points={len(xs)}"
        )

        # (a, b) = centro in body FLU (x avanti, y sinistra)
        # offset LiDAR rispetto al baricentro del drone (nel frame FRD)
        lidar_offset_frd = np.array([-0.10, 0.00, -0.26])  # attenzione: z è negativo perché FRD ha z verso il basso

        # centro della colonna rispetto al baricentro drone, in FRD
        c_frd = np.array([a, b, 0.0]) + lidar_offset_frd

        
        

        # Rotazione yaw NED (ψ assoluto) e traslazione con la posa del drone
        cpsi, spsi = math.cos(self.global_yaw), math.sin(self.global_yaw)

        Rz = np.array([[ cpsi, -spsi, 0.0],
                    [ spsi,  cpsi, 0.0],
                    [ 0.0 ,  0.0 , 1.0]])

        p_ned = np.array([self.global_x, self.global_y, self.global_z])  # posizione drone in NED

        c_ned = p_ned + (Rz @ c_frd)

        # Ora c_ned[:2] è il centro della colonna in coordinate globali NED (x Nord, y Est)
        self.get_logger().info(f"Centro in NED: x={c_ned[0]:.2f}, y={c_ned[1]:.2f}, R={R:.2f}")


        d_star = float(R + self.d_safe)

        # 2) vettori nel frame FRD
        d_c   = float(np.hypot(a, b))   # distanza planare centro (xy)

        # 3) fattibilità: serve spazio per stare a d*
       # if d_c <= d_star + self.delta_ok:
            # troppo vicino al centro per costruire il punto di stazionamento
        #    return None

        # 4) punto di stazionamento nel body FRD:
        #    p* = c - d* * (c/||c||)
        u = c_frd[:2] / d_c                      # versore orizzontale verso il centro
        p_star_frd_xy = c_frd[:2] - d_star * u   # solo xy
        p_star_frd = np.array([p_star_frd_xy[0], p_star_frd_xy[1], 0.0], dtype=float)

        # 5) yaw relativo (nel body FRD) che guarda il centro
        #    NB: lo yaw relativo è l'angolo del vettore p* in FRD
        psi_rel = math.atan2(p_star_frd[1], p_star_frd[0])

        # 6) trasformazione in NED: p*_NED = p_NED + Rz(psi_NED) * p*_FRD
        cpsi, spsi = math.cos(self.global_yaw), math.sin(self.global_yaw)
        
       
        p_star_ned = p_ned + (Rz @ p_star_frd)

        # 7) yaw assoluto del setpoint in NED
        psi_star = self._wrap_pi(self.global_yaw + psi_rel) #NORMALIZE YAW

        # 8) confeziona il setpoint (NED)
        
        self.current_position = {"x": p_star_ned[0], "y": p_star_ned[1], "z": self.global_z}
        self.yaw_target = psi_star

        # opzionale: log
        self.get_logger().info(
            f"Setpoint NED: x*={p_star_ned[0]:.2f}, y*={p_star_ned[1]:.2f}, z*={self.global_z:.2f}, yaw*={psi_star:.2f} rad  "
            f"(d*={R + self.d_safe:.2f} m)"
        )
            
        dx = p_star_ned[0] - self.global_x
        dy = p_star_ned[1] - self.global_y
        dist_err = math.hypot(dx, dy)

        if dist_err < 0.55 and self.phase != "CIRCULAR_PATH":
            self.get_logger().info("Approach complete → start CIRCULAR_PATH")
            self.phase = "CIRCULAR_PATH"
            self.column_center_ned = c_ned[:2]
            self.column_radius = R + self.d_safe

            # init phi from current pose to avoid cutting through the column
            cx, cy = self.column_center_ned
            self.phi = math.atan2(self.global_y - cy, self.global_x - cx)

            # time bookkeeping
            self._t_prev = self.get_clock().now().nanoseconds * 1e-9
            self.omega = 0.3          #6 rad/s
            self.delta_phi = 0.15      #35 rad lookahead on the circle
            self.wp_tol = 0.25
            return

        if self.phase == "CIRCULAR_PATH":
            cx, cy = self.column_center_ned
            r_star = self.column_radius

            # current target on the circle
            phi_sp = self.phi
            x_des = cx + r_star * math.cos(phi_sp)
            y_des = cy + r_star * math.sin(phi_sp)
            z_des = -5.0

            # yaw: always face the column center
            yaw_des = math.atan2(cy - y_des, cx - x_des)

            # set current target
            self.current_position = {"x": x_des, "y": y_des, "z": z_des}
            self.yaw_target = self._wrap_pi(yaw_des)

            # compute distance to current waypoint
            dx = x_des - self.global_x
            dy = y_des - self.global_y
            dist_err = math.hypot(dx, dy)

            # check if we reached it
            if dist_err < 0.25:
                # advance to the next point on the circle
                self.phi = self._wrap_pi(self.phi + self.delta_phi)
                self.get_logger().info(f"Reached WP, advancing φ → {math.degrees(self.phi):.1f}°")

            # optional safety: push back to circle if too close to center
            d_curr = math.hypot(self.global_x - cx, self.global_y - cy)
            if d_curr < r_star - 0.05:
                phi_now = math.atan2(self.global_y - cy, self.global_x - cx)
                x_des = cx + r_star * math.cos(phi_now)
                y_des = cy + r_star * math.sin(phi_now)
                self.current_position["x"] = x_des
                self.current_position["y"] = y_des
                self.get_logger().warn("Pushing outward — too close to column")

            # diagnostics
            if abs(d_curr - r_star) > 0.15:
                self.get_logger().warn(f"Range error: {d_curr:.2f} vs {r_star:.2f}")
       


    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True   # we want to control position
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_setpoint(self, coordinate:dict= None):
        sp = TrajectorySetpoint()
        #sp.timestamp = self._now_us()
     
        sp.position[0] = coordinate["x"]
        sp.position[1] = coordinate["y"]
        sp.position[2] = coordinate["z"]
        sp.yaw = self.yaw_target
        '''
        sp.velocity[0] = math.nan; sp.velocity[1] = math.nan; sp.velocity[2] = math.nan
        sp.acceleration[0] = math.nan; sp.acceleration[1] = math.nan; sp.acceleration[2] = math.nan
        sp.jerk[0] = math.nan; sp.jerk[1] = math.nan; sp.jerk[2] = math.nan
        #sp.yaw = math.nan
        sp.yawspeed = math.nan
        '''

        self.setpoint_pub.publish(sp)

    def arm(self):
        msg = VehicleCommand()
        #msg.timestamp = self._now_us()
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
    node = Trajectoryplanning()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
