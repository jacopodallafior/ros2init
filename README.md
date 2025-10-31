# 🧠 UAV Figure-8 PID Cascade Controller (Custom PX4 Offboard Control + Manual Mixer)

## ✈️ Overview
This project implements a **fully custom cascaded PID controller** that completely replaces the internal **PX4 position, attitude, and rate loops**.  
The entire control law runs in **ROS 2 Offboard mode**, sending **motor commands directly** via `ActuatorMotors`.

It forms a complete **end-to-end flight controller**:
- Generates 3D **Figure-8 trajectories**
- Runs **outer-loop PID** on position
- Runs **inner-loop PID** on attitude & angular rates**
- Allocates **thrust and torque → motor commands** through a **custom MMA (Manual Mixing Algorithm)**

This demonstrates full UAV control from **trajectory to motor PWM** entirely on the companion computer.

---

## ⚙️ Control Architecture

```
┌──────────────────────────────┐
│     Trajectory Generator     │
│     (Figure-8 Reference)     │
└──────────────┬───────────────┘
               │ (x_d, y_d, z_d, ψ_d)
┌──────────────┴───────────────┐
│     Outer Loop PID (Pos)     │
│ → Acceleration Command (a_d) │
└──────────────┬───────────────┘
               │ (a_x, a_y, a_z)
┌──────────────┴───────────────┐
│ Accel→Attitude Mapping       │
│ (a, ψ_d) → (φ_d, θ_d, T)     │
└──────────────┬───────────────┘
               │ (roll_d, pitch_d, yaw_d, thrust)
┌──────────────┴───────────────┐
│     Inner Loop PID (Att)     │
│ τ = Kp*e_rpy - Kd*ω + Ki∫e_rpy │
└──────────────┬───────────────┘
               │ (τx, τy, τz, T)
┌──────────────┴───────────────┐
│     MMA Mixer (4 motors)     │
│ → Actuator Commands [0–1]    │
└──────────────────────────────┘
```

---

## 🔧 Outer Loop: Position PID

Computes acceleration setpoints from position and velocity errors:

```
a_i = Kp_i * e_i + Ki_i * ∫e_i dt + Kd_i * de_i/dt
```

**Limits:**
- Horizontal acceleration: ±4 m/s²  
- Integrator cap: ±2.0  
- Integral leakage: 5 s  
- Output clamping and anti-windup logic.

**Example Gains**
| Axis | Kp | Ki | Kd |
|------|----|----|----|
| X | 0.3 | 0.01 | 0.75 |
| Y | 0.3 | 0.01 | 0.75 |
| Z | 0.65 | 0.02 | 0.70 |

---

## 🧭 Acceleration → Attitude Mapping

Uses a **geometric model** to compute roll, pitch, and thrust from desired accelerations:

```
pitch_d = atan2(fx*cos(ψ) + fy*sin(ψ), fz)
roll_d  = atan2(-fy*cos(ψ) + fx*sin(ψ), fz)
```
Where `fx = -ax`, `fy = -ay`, `fz = g - az`

**Tilt & thrust safety:**
- Tilt ≤ 45°  
- Normalized thrust ∈ [0.05, 0.9]

---

## 🔄 Inner Loop: Attitude PID (Body Frame)

Controls angular torque in FRD frame:

```
τ = Kp_eul * e_rpy - Kd_body * ω + Ki_eul * ∫e_rpy
```

**Example Gains:**
| Axis | Kp | Ki | Kd | Torque Max |
|------|----|----|----|-------------|
| Roll | 0.20 | 0.12 | 0.03 | ±0.15 |
| Pitch | 0.20 | 0.12 | 0.02 | ±0.15 |
| Yaw | 0.20 | 0.00 | 0.04 | ±0.15 |

Includes:
- Low-pass filtering of body rates (`fcut_rates = 30 Hz`)
- Integral clamping (`I_EUL_MAX = 0.3`)
- Bleed anti-windup during saturation
- Separate handling for yaw torque limitation

---

## ⚖️ Manual Mixing Algorithm (MMA)

Implements a **custom 4-motor allocation matrix** converting total thrust and body torques to per-motor normalized commands.

```
B = [
 [-y_i * kf],        # Mx
 [ x_i * kf],        # My
 [ s_i * km * kf],   # Mz
 [-kf]               # Fz
]
```

**Mixer steps:**
1. Compute least-squares motor outputs.  
2. Desaturate yaw first, then thrust.  
3. Apply **slew-rate limiting** (Δu ≤ 0.1 @ 400 Hz).  
4. Publish to `/fmu/in/actuator_motors`.

Built-in checks:
- `cond(B)` condition number  
- Roll/yaw direction verification  
- Hover wrench validation `B@u_hover ≈ [0,0,0,-m*g]`

---

## 🧰 Anti-Windup & Safety Logic

| Layer | Mechanism | Description |
|--------|------------|-------------|
| Position Z | Conditional + Clamping + Leakage | Stops integration when thrust saturates |
| Position XY | Gate + Leakage | Integrate only near steady state |
| Attitude | Clamping + Fast bleed | Zero I when torque saturated |
| Mixer | Slew-rate + desaturation | Avoids PWM spikes that trigger PX4 land detection |

---

## 🧱 Implementation Details

- **Language:** Python 3  
- **Framework:** ROS 2 (Foxy/Humble) + PX4-ROS2 Bridge  
- **Loop rate:** 400 Hz  
- **Direct actuator control:** `True`  
- **PX4 Topics Used:**

| Topic | Type | Role |
|-------|------|------|
| `/fmu/in/actuator_motors` | `ActuatorMotors` | Motor control |
| `/fmu/in/vehicle_thrust_setpoint` | `VehicleThrustSetpoint` | Publish thrust |
| `/fmu/in/vehicle_torque_setpoint` | `VehicleTorqueSetpoint` | Debug torque |
| `/fmu/out/vehicle_odometry` | `VehicleOdometry` | Attitude feedback |
| `/fmu/out/vehicle_local_position_v1` | `VehicleLocalPosition` | Position feedback |
| `/debug/prop[1-4]` | `Float32` | Individual motor debug |

---

## 🧪 Sanity Checks Before Flight

✅ `cond(B)` ≈ 100–150  
✅ `B @ u_hover ≈ [0, 0, 0, -m*g]`  
✅ `s·du_yaw > 0` (CCW motors increase with +yaw torque)  
✅ `(-propy)·du_roll > 0` (rear motors increase with +roll torque)  
✅ No actuator exceeds [0, 1] even at τ = ±0.15

---
