#!/usr/bin/env python3
import argparse, os, sys, yaml
import numpy as np
import matplotlib.pyplot as plt

# try 3D; fall back cleanly if unavailable
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_3D = True
except Exception:
    HAS_3D = False

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception as e:
    print("This script needs ROS 2 Python libs: rosbag2_py, rosidl_runtime_py")
    print("Install e.g.:  sudo apt install ros-<distro>-rosbag2-py ros-<distro>-rosidl-runtime-py")
    print("Detail:", e)
    sys.exit(1)

# ---------------- basic IO ----------------
def detect_storage_id(bag_dir: str) -> str:
    meta = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.exists(meta):
        return "sqlite3"
    with open(meta, "r") as f:
        y = yaml.safe_load(f)
    return y.get("rosbag2_bagfile_information", {}).get("storage_identifier", "sqlite3")

def get_type_map(reader):
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

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
    so = StorageOptions(uri=bag_dir, storage_id=detect_storage_id(bag_dir))
    co = ConverterOptions(input_serialization_format="", output_serialization_format="")
    r = SequentialReader(); r.open(so, co)
    type_map = get_type_map(r)
    if topic not in type_map:
        raise RuntimeError(f"Topic '{topic}' not found. Available: {list(type_map.keys())}")
    MsgType, ext = pick_point_extractor(type_map[topic])
    ts, xs, ys, zs = [], [], [], []
    while r.has_next():
        try: topic_name, data, t = r.read_next()
        except TypeError: t, data, topic_name = r.read_next()
        if topic_name != topic: continue
        msg = deserialize_message(data, MsgType)
        x, y, z = ext(msg)
        ts.append(t*1e-9); xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
    o = np.argsort(ts); ts = np.asarray(ts)[o]; xs = np.asarray(xs)[o]; ys = np.asarray(ys)[o]; zs = np.asarray(zs)[o]
    return ts, xs, ys, zs

def read_rpy(bag_dir, topic):
    so = StorageOptions(uri=bag_dir, storage_id=detect_storage_id(bag_dir))
    co = ConverterOptions(input_serialization_format="", output_serialization_format="")
    r = SequentialReader(); r.open(so, co)
    type_map = get_type_map(r)
    if topic not in type_map:
        raise RuntimeError(f"Topic '{topic}' not found. Available: {list(type_map.keys())}")
    MsgType, ext = pick_vec3_extractor(type_map[topic])
    ts, rr, pp, yy = [], [], [], []
    while r.has_next():
        try: topic_name, data, t = r.read_next()
        except TypeError: t, data, topic_name = r.read_next()
        if topic_name != topic: continue
        msg = deserialize_message(data, MsgType)
        x, y, z = ext(msg)
        ts.append(t*1e-9); rr.append(float(x)); pp.append(float(y)); yy.append(float(z))
    o = np.argsort(ts); ts = np.asarray(ts)[o]; rr = np.asarray(rr)[o]; pp = np.asarray(pp)[o]; yy = np.asarray(yy)[o]
    return ts, rr, pp, yy

# ---------------- metrics helpers ----------------
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def overlapped_range(t1, t2, skip_sec=0.0):
    tmin = max(t1[0], t2[0]) + float(skip_sec)
    tmax = min(t1[-1], t2[-1])
    return (tmin, tmax) if tmax > tmin else (None, None)

def resample(t_ref, v_ref, t_act, v_act, skip_sec=0.0, n=3000):
    tmin, tmax = overlapped_range(t_ref, t_act, skip_sec)
    if tmin is None: return None, None, None
    tt = np.linspace(tmin, tmax, n)
    r = np.interp(tt, t_ref, v_ref)
    a = np.interp(tt, t_act, v_act)
    return tt, r, a

def estimate_lag(tt, r, a):
    r0 = r - np.mean(r); a0 = a - np.mean(a)
    corr = np.correlate(a0, r0, mode="full")   # shift a relative to r
    lag_idx = corr.argmax() - (len(a0) - 1)
    dt = lag_idx * (tt[1]-tt[0])
    return dt

def set_equal_3d(ax, x, y, z):
    xr = np.max(x)-np.min(x); yr = np.max(y)-np.min(y); zr = np.max(z)-np.min(z)
    r = max(xr, yr, zr, 1.0)
    cx = 0.5*(np.max(x)+np.min(x)); cy = 0.5*(np.max(y)+np.min(y)); cz = 0.5*(np.max(z)+np.min(z))
    ax.set_box_aspect((1,1,1))
    ax.set_xlim(cx-r/2, cx+r/2); ax.set_ylim(cy-r/2, cy+r/2); ax.set_zlim(cz-r/2, cz+r/2)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Plot XY + 3D + RPY (with lag- and skip-aware RMS) from a rosbag2 directory.")
    ap.add_argument("bag")
    ap.add_argument("--ref", default="/debug/ref_xyz")
    ap.add_argument("--act", default="/debug/xyz")
    ap.add_argument("--rpy-ref", default="/debug/rpy_ref")
    ap.add_argument("--rpy-act", default="/debug/rpy")
    ap.add_argument("--skip-sec", type=float, default=0.0, help="ignore first N seconds when computing errors")
    ap.add_argument("--save", default=None)
    ap.add_argument("--save3d", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.bag) or not os.path.exists(os.path.join(args.bag, "metadata.yaml")):
        print("ERROR: not a rosbag2 directory:", args.bag); sys.exit(2)

    # --- XYZ ---
    t_r, xr, yr, zr = read_xyz(args.bag, args.ref)
    t_a, xa, ya, za = read_xyz(args.bag, args.act)

    # XY overlay
    plt.figure()
    plt.plot(xr, yr, "--", label="reference")
    plt.plot(xa, ya, "-",  label="actual")
    plt.axis("equal"); plt.grid(True)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.legend(); plt.title("XY ground track: reference vs actual")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=180); print("Saved:", args.save)

    # RMS XY (plain + lag-corrected), after skip
    out = resample(t_r, xr, t_a, xa, skip_sec=args.skip_sec); 
    if out is not None:
        tt, xr_i, xa_i = out
        _, yr_i, ya_i = resample(t_r, yr, t_a, ya, skip_sec=args.skip_sec)
        ex = xa_i - xr_i; ey = ya_i - yr_i
        rms_xy = np.sqrt(np.mean(ex**2 + ey**2))
        print(f"RMS XY (skip {args.skip_sec:.1f}s, no lag corr): {rms_xy:.3f}")

        # lag per axis and lag-corrected RMS
        dt_x = estimate_lag(tt, xr_i, xa_i)
        dt_y = estimate_lag(tt, yr_i, ya_i)
        dt = 0.5*(dt_x + dt_y)
        xr_s = np.interp(tt, t_r+dt, xr); yr_s = np.interp(tt, t_r+dt, yr)
        ex2 = xa_i - xr_s; ey2 = ya_i - yr_s
        rms_xy_lag = np.sqrt(np.mean(ex2**2 + ey2**2))
        print(f"Estimated lag dt_x={dt_x:.3f}s dt_y={dt_y:.3f}s  → avg dt={dt:.3f}s")
        print(f"RMS XY (skip {args.skip_sec:.1f}s, lag-corrected): {rms_xy_lag:.3f}")

        # time plots
        plt.figure(figsize=(12,5))
        plt.plot(tt, xr_i, "--", label="x_ref"); plt.plot(tt, xa_i, "-", label="x")
        plt.plot(tt, yr_i, "--", label="y_ref"); plt.plot(tt, ya_i, "-", label="y")
        plt.grid(True); plt.xlabel("t [s]"); plt.ylabel("position")
        plt.title("Position vs reference"); plt.legend(); plt.tight_layout()

    # 3D trajectory
    if HAS_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xr, yr, zr, "--", label="reference")
        ax.plot(xa, ya, za, "-",  label="actual")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("3D trajectory: reference vs actual")
        set_equal_3d(ax, np.r_[xr,xa], np.r_[yr,ya], np.r_[zr,za])
        ax.legend()
        if args.save3d:
            plt.savefig(args.save3d, dpi=180); print("Saved 3D:", args.save3d)
    else:
        print("3D backend not available; skipping 3D plot.")

    # --- RPY with RMS ---
    try:
        tr_ref, roll_ref, pitch_ref, yaw_ref = read_rpy(args.bag, args.rpy_ref)
        tr_act, roll_act, pitch_act, yaw_act = read_rpy(args.bag, args.rpy_act)

        out_r = resample(tr_ref, roll_ref, tr_act, roll_act, skip_sec=args.skip_sec, n=4000)
        out_p = resample(tr_ref, pitch_ref, tr_act, pitch_act, skip_sec=args.skip_sec, n=4000)
        out_y = resample(tr_ref, yaw_ref,  tr_act, yaw_act,  skip_sec=args.skip_sec, n=4000)
        if out_r and out_p and out_y:
            ttr, r_ref_i, r_act_i = out_r
            _,   p_ref_i, p_act_i = out_p
            _,   y_ref_i, y_act_i = out_y

            er = r_act_i - r_ref_i
            ep = p_act_i - p_ref_i
            eyaw = (y_act_i - y_ref_i + np.pi) % (2*np.pi) - np.pi
            rms_r = np.sqrt(np.mean(er**2))
            rms_p = np.sqrt(np.mean(ep**2))
            rms_y = np.sqrt(np.mean(eyaw**2))
            print(f"RMS roll  [rad]: {rms_r:.4f}  [{np.degrees(rms_r):.2f} deg]")
            print(f"RMS pitch [rad]: {rms_p:.4f}  [{np.degrees(rms_p):.2f} deg]")
            print(f"RMS yaw   [rad]: {rms_y:.4f}  [{np.degrees(rms_y):.2f} deg]")

            fig2, axr = plt.subplots(3, 1, figsize=(12,6), sharex=True)
            axr[0].plot(ttr, r_ref_i, "--", label="roll_ref");  axr[0].plot(ttr, r_act_i, "-", label="roll")
            axr[1].plot(ttr, p_ref_i, "--", label="pitch_ref"); axr[1].plot(ttr, p_act_i, "-", label="pitch")
            axr[2].plot(ttr, y_ref_i, "--", label="yaw_ref");   axr[2].plot(ttr, y_act_i, "-", label="yaw")
            for a in axr: a.grid(True)
            axr[0].set_ylabel("roll [rad]"); axr[1].set_ylabel("pitch [rad]"); axr[2].set_ylabel("yaw [rad]")
            axr[2].set_xlabel("t [s]")
            axr[0].legend(); axr[1].legend(); axr[2].legend()
            fig2.suptitle("Attitude vs reference"); fig2.tight_layout(rect=[0,0.03,1,0.97])
    except RuntimeError as e:
        print("RPY read skipped:", e)

    plt.show()

if __name__ == "__main__":
    main()
