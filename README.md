# 🧠 UAV Figure-8 PID Cascade Controller (Custom PX4 Offboard Control)

## ✈️ Overview

This project implements a **fully custom cascaded PID controller** that replaces the default **PX4 attitude, rate, and position controllers**, running entirely in **ROS 2 Offboard mode**.  
It commands **thrust and torque setpoints** (`VehicleThrustSetpoint` and `VehicleTorqueSetpoint`) directly to PX4 — effectively acting as an external flight control law.

The system executes a **Figure-8 (∞)** trajectory in 3D space using a **manual multi-axis (MMA)** control strategy:
- **Outer loop:** Position control (PID on x, y, z)
- **Inner loop:** Attitude and rate control (PD + I damping in body frame)
- **Mixer equivalent:** Converts thrust/torque setpoints into normalized PX4 actuator commands

This setup demonstrates full-stack control — from trajectory generation to body torque — without relying on PX4’s internal cascades.

---
## 🔧 Control Logic

### 🧩 Position PID (Outer Loop)
- Inputs: drone position `(x, y, z)` and reference `(x_d, y_d, z_d)`
- Outputs: acceleration commands `(a_x, a_y, a_z)`
- Equation:
a_i = Kp_i * e_i + Ki_i * ∫e_i dt + Kd_i * de_i/dt

- Limits:
- Horizontal accel saturation: ±6 m/s²
- Integrator cap: ±2.0
- Integral leakage: 5 s time constant

### 🎯 Accel → Attitude Mapping
Converts desired accelerations and yaw into target roll, pitch, and normalized thrust using a geometric model:

pitch_d = atan2( fxcosψ + fysinψ, fz )
roll_d = atan2( -fycosψ + fxsinψ, fz )
thrust_norm = |f| / g


Safety:
- Tilt limited to ±45°.
- Thrust clamped between 0.05 – 0.9 (normalized PX4 range).

### 🔄 Attitude PID (Inner Loop)
Operates in **Euler angle space** with damping on **body rates**:

τ = Kp_eul * e_rpy - Kd_body * ω + Ki_eul * ∫e_rpy dt

where `ω = [p, q, r]`.

**Typical gains:**

| Axis | Kp | Ki | Kd | Max Torque |
|------|----|----|----|-------------|
| Roll | 0.75 | 0.12 | 0.10 | ±0.15 |
| Pitch | 0.75 | 0.12 | 0.10 | ±0.15 |
| Yaw | 0.35 | 0.00 | 0.06 | ±0.15 |

These were tuned to achieve smooth yet responsive behavior during the figure-8 maneuver.

---

## 🌀 Figure-8 Trajectory Generator

Implements a **Lissajous curve** to generate a continuous infinity symbol in the XY-plane:

\[
x(t) = x_0 + A_x \sin(\omega t)
\]
\[
y(t) = y_0 + A_y \sin(2\omega t + \phi)
\]

Parameters:
- `Ax = 40 m`, `Ay = 40 m`
- `period = 30 s`
- `phase = 0 rad`

Yaw can either:
- **Follow tangent** to the trajectory (`follow_tangent_yaw = True`)
- **Stay fixed forward** (`False`)

Yaw tracking uses a “gate” (10° default) to prevent sharp discontinuities — the reference is frozen until actual yaw catches up.

---

## 🧱 Implementation Details

- **Language:** Python 3 (ROS 2 Foxy + PX4 ROS 2 Bridge)
- **Messages used:**
  - `VehicleThrustSetpoint`
  - `VehicleTorqueSetpoint`
  - `VehicleOdometry`
  - `VehicleLocalPosition`
  - `OffboardControlMode`
  - `VehicleCommand`
- **Execution rate:** 125 Hz (0.008 s loop)
- **Offboard control:** activated automatically after arming.

---

## 🧩 Topics

| Topic | Type | Role |
|-------|------|------|
| `/fmu/in/vehicle_torque_setpoint` | `px4_msgs/msg/VehicleTorqueSetpoint` | Sends normalized torque commands |
| `/fmu/in/vehicle_thrust_setpoint` | `px4_msgs/msg/VehicleThrustSetpoint` | Sends normalized thrust |
| `/fmu/out/vehicle_odometry` | `px4_msgs/msg/VehicleOdometry` | Provides orientation and angular velocity |
| `/fmu/out/vehicle_local_position_v1` | `px4_msgs/msg/VehicleLocalPosition` | Provides position and velocity |
| `/debug/torquex`, `/torquey`, `/torquez` | `std_msgs/msg/Float32` | Debug torque outputs |

---

## 📊 Debugging and Tuning

Visualize control effort with:
```bash
rqt_plot /debug/torquex/data /debug/torquey/data /debug/torquez/data
