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
        super().__init__('Trajectoryplanning')
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
        self.object_fixed = None
        #self.pose_ready = False
        #self.sent_setpoints = 0
        #self.offboard_enabled = False

     
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
         
    def _now_us(self) -> int:
        return self.get_clock().now().nanoseconds // 1000


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

        # index of nearest hit
        i0 = int(np.argmin(ranges_v))
        r0 = ranges_v[i0]

        # grow a small contiguous cluster around i0 (range-threshold and index continuity)
        dr_thresh = 0.25   # meters
        left = i0
        while left-1 >= 0 and abs(ranges_v[left-1] - r0) < dr_thresh and (i0 - (left-1)) < 50:
            left -= 1
        right = i0
        N = len(ranges_v)
        while right+1 < N and abs(ranges_v[right+1] - r0) < dr_thresh and ((right+1) - i0) < 50:
            right += 1

        r_cluster = ranges_v[left:right+1]
        a_cluster = angles_v[left:right+1]
        if r_cluster.size < 3:
            # too few points, skip this frame
            return

        # body-aligned LiDAR (SDF has zero rotation), 2D Cartesian
        xs = r_cluster * np.cos(a_cluster)
        ys =-( r_cluster * np.sin(a_cluster) )# CONVENTion

        centroid_x_local = float(np.mean(xs))
        centroid_y_local = float(np.mean(ys))  

        # ---------- 2) body → global (PX4 NED rotation) ----------
        c = math.cos(self.global_yaw)
        s = math.sin(self.global_yaw)

        # optional small physical offset (from your SDF)
        xB = centroid_x_local - 0.10
        yB = centroid_y_local - 0.00
        # z offset is irrelevant for 2D approach

        # NED rotation: [xN; yE] = [[ c, s], [-s, c]] * [xB; yB]
        xN = c * xB + -s * yB
        yE = s * xB + c * yB

        centroid_x_global = self.global_x + xN
        centroid_y_global = self.global_y + yE

        # ---------- 3) lock object once (first stable detection) ----------
        if self.object_fixed is None:
            self.object_fixed = {"x": centroid_x_global, "y": centroid_y_global}
            self.get_logger().info(f"Object fixed at x={centroid_x_global:.2f}, y={centroid_y_global:.2f}")

            # compute a single approach target with stand-off
            dx = self.object_fixed["x"] - self.global_x
            dy = self.object_fixed["y"] - self.global_y
            self.D = math.hypot(dx, dy)
            print(f"la distanza dall'oggetto è {self.D}")


            desired_distance = 1.0  # <-- set your stand-off here

            if self.D > desired_distance + 0.05:
                factor = (self.D - desired_distance) / self.D
                target_x = self.global_x + dx * factor
                target_y = self.global_y + dy * factor
            else:
                target_x, target_y = self.global_x, self.global_y

            self.approach_target = {"x": target_x, "y": target_y, "z": target_altitude}
            # start moving toward it (publish loop will smooth)
            self.current_position = self.approach_target.copy()

            self.get_logger().info(
                f"Locked single target: ({target_x:.2f}, {target_y:.2f}, {target_altitude:.2f}) "
                f"(stand-off {desired_distance:.1f} m)"
            
            )
            return
        #self.object_fixed = None

        # ---------- 4) keep commanding the same fixed target ----------
        if self.approach_target is not None:
            self.current_position = self.approach_target
            print(f"la posizione è {self.current_position}")
            print(f"Detected object at global x={centroid_x_global:.2f}, y={centroid_y_global:.2f}")
            print(f"la distanza da drone è {self.D}")

    
    def lidar_callback2(self, msg:LaserScan):

        if not hasattr(self, "phase"):
            self.phase = "TAKEOFF"

        # Desired altitude (NED, negative up)
        target_altitude = -5.0

        if self.phase == "TAKEOFF":
            # Check vertical error only
            altitude_error = abs(self.global_z - target_altitude)
            if altitude_error > 0.3:
                # Keep climbing/descending vertically
                self.current_position = {"x": self.global_x, "y": self.global_y, "z": target_altitude}
                self.get_logger().info(f"Climbing to altitude {target_altitude:.1f} (current {self.global_z:.2f})")
            else:
                # Once altitude reached, switch to approach phase
                self.phase = "APPROACH"
                self.get_logger().info("Altitude reached → switching to APPROACH phase")
            return 
                
        """Detect nearest object and compute its global position."""

        # Convert full scan to numpy arrays
        ranges = np.array(msg.ranges)
        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(len(ranges))])

        # --- Filter out invalid or too-far readings ---
        valid_mask = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        if len(ranges) == 0:
            self.get_logger().info("No valid LiDAR returns.")
            return

        # --- Convert to Cartesian (LiDAR/body frame) ---
        xs = ranges * np.cos(angles)
        ys = -(ranges * np.sin(angles))

        # --- Simple nearest-object detection: pick points within small radius of the closest one ---
        
        # Centroid of cluster in body frame
        centroid_x_local = np.mean(xs)
        centroid_y_local = np.mean(ys)

        # --- Transform centroid to global frame ---  #NED MATH ROTATION
        #yaw = getattr(self, "global_yaw", 0.0)
        
        
        lidar_offset_body = np.array([-0.10, 0.0, 0.26])     # LIDAR OFFSET in metres
        rot = np.array([[math.cos(self.global_yaw),  math.sin(self.global_yaw), 0],
                        [-math.sin(self.global_yaw), math.cos(self.global_yaw), 0],
                        [0, 0, 1]])                           # NED rotation
        offset_global = rot @ lidar_offset_body
        centroid_x_global = self.global_x + offset_global[0] + (
            centroid_x_local * math.cos(self.global_yaw) - centroid_y_local * math.sin(self.global_yaw))
        centroid_y_global = self.global_y + offset_global[1] + (
        + centroid_x_local * math.sin(self.global_yaw) + centroid_y_local * math.cos(self.global_yaw))
        centroid_z_global = self.global_z + offset_global[2]

        self.get_logger().info(
            f"yaw={self.global_yaw:.2f} rad | centroid_local=({centroid_x_local:.2f},{centroid_y_local:.2f}) "
            f"→ global=({centroid_x_global:.2f},{centroid_y_global:.2f})"
        )


        # --- Store and print result ---
        self.object_global = {
            "x": centroid_x_global,
            "y": centroid_y_global,
            "z": centroid_z_global,
        }
        self.get_logger().info(
            f"Detected object at global x={centroid_x_global:.2f}, y={centroid_y_global:.2f}"
        )

        # --- After computing self.object_global ---
        if self.phase == "APPROACH":
            dx = self.object_global["x"] - self.global_x
            dy = self.object_global["y"] - self.global_y
            distance = math.hypot(dx, dy)
            self.desired_yaw = math.atan2(dy, dx)

            print(f"la distanza da drone è {distance}")


            # Desired standoff distance (1 m)
           
            desired_distance = 5.0
            

            if distance > desired_distance + 0.3: #and self.num == 1:
                # Compute a point desired_distance away from object, along the line of sight
                factor = (distance - desired_distance) / distance
                target_x = self.global_x + dx #* factor
                target_y = self.global_y + dy #* factor
                target_z = target_altitude  # maintain altitude
                self.num +=1 

                self.current_position = {"x": target_x, "y": target_y, "z": target_z}
                self.get_logger().info(
                    f"Approaching object: target=({target_x:.2f},{target_y:.2f}) dist={distance:.2f}"
                )
            else:
                # Arrived at the desired standoff distance
                self.get_logger().info("Reached 1 m from object — holding position.")
               # self.get_logger().info(f"Approaching object: target=({target_x:.2f},{target_y:.2f}) dist={distance:.2f}")
                #self.phase = "HOLD"

                

        '''
        # RECIVE DATA FROM THE LIDAR


        self.lower_anglimit = 1,39626
        self.upper_anglimit = 1,74533
        
        i_start = int((self.lower_anglimit - msg.angle_min) / msg.angle_increment)
        i_end   = int((self.upper_anglimit - msg.angle_min) / msg.angle_increment)

# Extract only that slice of data
        ranges_subset = msg.ranges[i_start:i_end]
        angles_subset = [msg.angle_min + i * msg.angle_increment for i in range(i_start, i_end)]

        #for a, r in zip(angles_subset, ranges_subset):
         #   print(f"Angle: {np.degrees(a):6.2f}°, Distance: {r:.2f} m")

    


        # CENTROID CONTROL
        valid_mask = np.isfinite(ranges_subset)
        ranges_valid = ranges_subset[valid_mask]
        angles_valid = angles_subset[valid_mask]

        if len(ranges_valid) == 0:
            self.get_logger().info("No valid LiDAR points in 80°– 100° sector")
            return

        # Convert to Cartesian (LiDAR/body frame)
        x_local = ranges_valid * np.cos(angles_valid)
        y_local = ranges_valid * np.sin(angles_valid)

        # Compute centroid in local frame
        centroid_x_local = np.mean(x_local)
        centroid_y_local = np.mean(y_local)

        # --- Transform to global frame ---
        # Rotation by drone’s yaw (heading)
        yaw = getattr(self, 'global_yaw', 0.0)
        centroid_x_global = self.global_x + (centroid_x_local * np.cos(yaw) - centroid_y_local * np.sin(yaw))
        centroid_y_global = self.global_y + (centroid_x_local * np.sin(yaw) + centroid_y_local * np.cos(yaw))

        # (z stays the same height as the LiDAR)
        centroid_z_global = self.global_z

        # Store or publish result
        self.centroid_global = {
            "x": centroid_x_global,
            "y": centroid_y_global,
            "z": centroid_z_global
        }

        self.get_logger().info(
            f"LiDAR centroid (global): x={centroid_x_global:.2f}, y={centroid_y_global:.2f}, z={centroid_z_global:.2f}"
        )

        # next point calculation 


        # NEXT POINT POSITION
        
        self.reftraj.append({"x": xpos, "y": ypos, "z": -5.0})

        if (abs(self.error_x) < 0.5 and abs(self.error_y) < 0.5 and abs(self.error_z) < 0.5):
            if not hasattr(self, "inside_since"):
                self.inside_since = time.time()   # just entered
            elif time.time() - self.inside_since > 1.0:  # stayed 1s
               self.refcount += 1
            if self.refcount < len(self.reftraj):
                    self.current_position = self.reftraj[self.refcount]
                    print("POSIZIONE RAGGIUNTA")
        else:
     # reset timer if you go out of tolerance
            if hasattr(self, "inside_since"):
                del self.inside_since
        self.get_logger().info(f"Received scan with {len(msg.ranges)} points")

        '''
        

    def publish_offboard_mode(self):
        msg = OffboardControlMode()
       # msg.timestamp = self._now_us()
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
        sp.yaw = 0.0#self.desired_yaw
        '''
        sp.velocity[0] = math.nan; sp.velocity[1] = math.nan; sp.velocity[2] = math.nan
        sp.acceleration[0] = math.nan; sp.acceleration[1] = math.nan; sp.acceleration[2] = math.nan
        sp.jerk[0] = math.nan; sp.jerk[1] = math.nan; sp.jerk[2] = math.nan
        #sp.yaw = math.nan
        sp.yawspeed = math.nan
        '''

 # z (NED, so -10 = climb up 10 m)
        #sp.yaw = 0.0
        self.setpoint_pub.publish(sp)

        #self.sent_setpoints += 1

    # Switch to Offboard only once: after the stream is established and pose is ready
       # if (not self.offboard_enabled) and self.pose_ready and self.sent_setpoints > 10:
        #    self.get_logger().info("Conditions met → switching to OFFBOARD now")
         #   self.set_offboard_mode()
          #  self.offboard_enabled = True

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

       # if not self.pose_ready:
        #    self.get_logger().warn("No VehicleLocalPosition yet; not switching to OFFBOARD")
         #   return
        #if self.sent_setpoints <= 10:
         #   self.get_logger().warn("Not enough setpoints streamed yet; not switching to OFFBOARD")
          #  return
        msg = VehicleCommand()
     #   msg.timestamp = self._now_us() 
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
       # self.current_position = {"x": 0.0, "y": 0.0, "z": -5.0}


def main(args=None):
    rclpy.init(args=args)
    node = Trajectoryplanning()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
