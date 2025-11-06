#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

# optional 3D (graceful fallback if missing)
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_3D = True
except Exception:
    HAS_3D = False

# Requires: rosbag2_py, rosidl_runtime_py
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception as e:
    print("This script needs ROS 2 Python libs: rosbag2_py, rosidl_runtime_py")
    print("Install e.g.:  sudo apt install ros-<distro>-rosbag2-py ros-<distro>-rosidl-runtime-py")
    print("Detail:", e)
    sys.exit(1)

def detect_storage_id(bag_dir: str) -> str:
    meta = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.exists(meta):
        return "sqlite3"
    with open(meta, "r") as f:
        y = yaml.safe_load(f)
    return y.get("rosbag2_bagfile_information", {}).get("storage_identifier", "sqlite3")

def get_type_map(reader):
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

# -------- Readers --------

def pick_point_extractor(ros_type):
    MsgType = get_message(ros_type)
    if ros_type.endswith("geometry_msgs/msg/Point"):
        def ext(m): return (m.x, m.y, m.z)
        return MsgType, ext
    if ros_type.endswith("geometry_msgs/msg/PointStamped"):
        def ext(m): return (m.point.x, m.point.y, m.point.z)
        return MsgType, ext
    raise RuntimeError(f"Unsupported type {ros_type}; need Point or PointStamped.")

def pick_vec3_extractor(ros_type):
    MsgType = get_message(ros_type)
    if ros_type.endswith("geometry_msgs/msg/Vector3"):
        def ext(m): return (m.x, m.y, m.z)
        return MsgType, ext
    if ros_type.endswith("geometry_msgs/msg/Vector3Stamped"):
        def ext(m): return (m.vector.x, m.vector.y, m.vector.z)
        return MsgType, ext
    raise RuntimeError(f"Unsupported type {ros_type}; need Vector3 or Vector3Stamped.")

def read_xyz(bag_dir, topic):
    storage_id = detect_storage_id(bag_dir)
    so = StorageOptions(uri=bag_dir, storage_id=storage_id)
    co = ConverterOptions(input_serialization_format="", output_serialization_format="")
    r = SequentialReader(); r.open(so, co)

    type_map = get_type_map(r)
    if topic not in type_map:
        raise RuntimeError(f"Topic '{topic}' not found. Available: {list(type_map.keys())}")
    MsgType, ext = pick_point_extractor(type_map[topic])

    ts, xs, ys, zs = [], [], [], []
    while r.has_next():
        try:
            topic_name, data, t = r.read_next()
        except TypeError:
            t, data, topic_name = r.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(data, MsgType)
        x, y, z = ext(msg)
        ts.append(t * 1e-9); xs.append(float(x)); ys.append(float(y)); zs.append(float(z))

    order = np.argsort(ts)
    return np.asarray(ts)[order], np.asarray(xs)[order], np.asarray(ys)[order], np.asarray(zs)[order]

def read_rpy(bag_dir, topic):
    storage_id = detect_storage_id(bag_dir)
    so = StorageOptions(uri=bag_dir, storage_id=storage_id)
    co = ConverterOptions(input_serialization_format="", output_serialization_format="")
    r = SequentialReader(); r.open(so, co)

    type_map = get_type_map(r)
    if topic not in type_map:
        raise RuntimeError(f"Topic '{topic}' not found. Available: {list(type_map.keys())}")
    MsgType, ext = pick_vec3_extractor(type_map[topic])

    ts, roll, pitch, yaw = [], [], [], []
    while r.has_next():
        try:
            topic_name, data, t = r.read_next()
        except TypeError:
            t, data, topic_name = r.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(data, MsgType)
        x, y, z = ext(msg)  # roll, pitch, yaw (rad)
        ts.append(t * 1e-9); roll.append(float(x)); pitch.append(float(y)); yaw.append(float(z))

    order = np.argsort(ts)
    return (np.asarray(ts)[order],
            np.asarray(roll)[order],
            np.asarray(pitch)[order],
            np.asarray(yaw)[order])

# -------- Helpers --------

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def set_equal_3d(ax, x, y, z):
    # Equal aspect for 3D
    xr = np.max(x) - np.min(x)
    yr = np.max(y) - np.min(y)
    zr = np.max(z) - np.min(z)
    r = max(xr, yr, zr)
    if r <= 0:
        r = 1.0
    cx = (np.max(x) + np.min(x)) / 2.0
    cy = (np.max(y) + np.min(y)) / 2.0
    cz = (np.max(z) + np.min(z)) / 2.0
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(cx - r/2, cx + r/2)
    ax.set_ylim(cy - r/2, cy + r/2)
    ax.set_zlim(cz - r/2, cz + r/2)

# -------- Main --------

def main():
    ap = argparse.ArgumentParser(description="Plot XY + 3D + RPY reference vs actual from a rosbag2 directory.")
    ap.add_argument("bag", help="Path to rosbag2 directory (contains metadata.yaml)")
    ap.add_argument("--ref", default="/debug/ref_xyz", help="Reference position topic (Point or PointStamped)")
    ap.add_argument("--act", default="/debug/xyz", help="Actual position topic (Point or PointStamped)")
    ap.add_argument("--rpy-ref", default="/debug/rpy_ref", help="Reference RPY topic (Vector3 or Vector3Stamped)")
    ap.add_argument("--rpy-act", default="/debug/rpy", help="Actual RPY topic (Vector3 or Vector3Stamped)")
    ap.add_argument("--save", default=None, help="Optional PNG path for XY plot")
    ap.add_argument("--save3d", default=None, help="Optional PNG path for 3D plot")
    args = ap.parse_args()

    if not os.path.isdir(args.bag) or not os.path.exists(os.path.join(args.bag, "metadata.yaml")):
        print("ERROR: not a rosbag2 directory:", args.bag); sys.exit(2)

    # --- XYZ ---
    t_r, xr, yr, zr = read_xyz(args.bag, args.ref)
    t_a, xa, ya, za = read_xyz(args.bag, args.act)

    # 2D XY overlay
    plt.figure()
    plt.plot(xr, yr, "--", label="reference")
    plt.plot(xa, ya, "-",  label="actual")
    plt.axis("equal"); plt.grid(True)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.legend(); plt.title("XY ground track: reference vs actual")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=180); print("Saved:", args.save)

    # Interpolate to common grid for RMS XY
    t0 = max(t_r[0], t_a[0]); t1 = min(t_r[-1], t_a[-1])
    if t1 > t0:
        tt = np.linspace(t0, t1, 3000)
        xr_i = np.interp(tt, t_r, xr); xa_i = np.interp(tt, t_a, xa)
        yr_i = np.interp(tt, t_r, yr); ya_i = np.interp(tt, t_a, ya)
        ex = xa_i - xr_i; ey = ya_i - yr_i
        rms_xy = np.sqrt(np.mean(ex**2 + ey**2))
        print(f"RMS XY error over overlap: {rms_xy:.3f}")

        plt.figure()
        plt.plot(tt, xr_i, "--", label="x_ref"); plt.plot(tt, xa_i, "-", label="x")
        plt.plot(tt, yr_i, "--", label="y_ref"); plt.plot(tt, ya_i, "-", label="y")
        plt.grid(True); plt.xlabel("t [s]"); plt.ylabel("position")
        plt.title("Position vs reference"); plt.legend(); plt.tight_layout()

    # 3D plot (if backend available)
    if HAS_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xr, yr, zr, "--", label="reference")
        ax.plot(xa, ya, za, "-",  label="actual")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("3D trajectory: reference vs actual")
        set_equal_3d(ax, np.concatenate([xr, xa]), np.concatenate([yr, ya]), np.concatenate([zr, za]))
        ax.legend()
        if args.save3d:
            plt.savefig(args.save3d, dpi=180); print("Saved 3D:", args.save3d)
    else:
        print("3D backend not available; skipping 3D plot. "
              "If you want it: sudo apt install python3-matplotlib && avoid mixed pip/apt installs.")

    # --- RPY ---
    try:
        tr_ref, roll_ref, pitch_ref, yaw_ref = read_rpy(args.bag, args.rpy_ref)
        tr_act, roll_act, pitch_act, yaw_act = read_rpy(args.bag, args.rpy_act)

        t0r = max(tr_ref[0], tr_act[0]); t1r = min(tr_ref[-1], tr_act[-1])
        if t1r > t0r:
            ttr = np.linspace(t0r, t1r, 4000)
            r_ref_i = np.interp(ttr, tr_ref, roll_ref); r_act_i = np.interp(ttr, tr_act, roll_act)
            p_ref_i = np.interp(ttr, tr_ref, pitch_ref); p_act_i = np.interp(ttr, tr_act, pitch_act)
            y_ref_i = np.interp(ttr, tr_ref, yaw_ref);   y_act_i = np.interp(ttr, tr_act, yaw_act)

            er = r_act_i - r_ref_i
            ep = p_act_i - p_ref_i
            eyaw = (y_act_i - y_ref_i + np.pi) % (2*np.pi) - np.pi  # wrap

            rms_r = np.sqrt(np.mean(er**2))
            rms_p = np.sqrt(np.mean(ep**2))
            rms_y = np.sqrt(np.mean(eyaw**2))
            print(f"RMS roll  [rad]: {rms_r:.4f}  [{np.degrees(rms_r):.2f} deg]")
            print(f"RMS pitch [rad]: {rms_p:.4f}  [{np.degrees(rms_p):.2f} deg]")
            print(f"RMS yaw   [rad]: {rms_y:.4f}  [{np.degrees(rms_y):.2f} deg]")

            fig2, axr = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axr[0].plot(ttr, r_ref_i, "--", label="roll_ref");  axr[0].plot(ttr, r_act_i, "-", label="roll")
            axr[1].plot(ttr, p_ref_i, "--", label="pitch_ref"); axr[1].plot(ttr, p_act_i, "-", label="pitch")
            axr[2].plot(ttr, y_ref_i, "--", label="yaw_ref");   axr[2].plot(ttr, y_act_i, "-", label="yaw")
            for a in axr: a.grid(True)
            axr[0].set_ylabel("roll [rad]"); axr[1].set_ylabel("pitch [rad]"); axr[2].set_ylabel("yaw [rad]")
            axr[2].set_xlabel("t [s]")
            axr[0].legend(); axr[1].legend(); axr[2].legend()
            fig2.suptitle("Attitude vs reference")
            fig2.tight_layout(rect=[0, 0.03, 1, 0.97])
        else:
            print("Not enough RPY time overlap to plot.")
    except RuntimeError as e:
        print("RPY read skipped:", e)

    plt.show()

if __name__ == "__main__":
    main()
