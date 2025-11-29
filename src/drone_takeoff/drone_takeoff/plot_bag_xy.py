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

MIN_THRUST = 0.05   # same normalization as in your controller
MAX_THRUST = 1.0
SAT_EPS    = 0.01   # small tolerance to detect sat numerically

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
        try:
            topic_name, data, t = r.read_next()
        except TypeError:
            t, data, topic_name = r.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(data, MsgType)
        x, y, z = ext(msg)
        ts.append(t * 1e-9)
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
    o = np.argsort(ts)
    ts = np.asarray(ts)[o]
    xs = np.asarray(xs)[o]
    ys = np.asarray(ys)[o]
    zs = np.asarray(zs)[o]
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
        try:
            topic_name, data, t = r.read_next()
        except TypeError:
            t, data, topic_name = r.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(data, MsgType)
        x, y, z = ext(msg)
        ts.append(t * 1e-9)
        rr.append(float(x))
        pp.append(float(y))
        yy.append(float(z))
    o = np.argsort(ts)
    ts = np.asarray(ts)[o]
    rr = np.asarray(rr)[o]
    pp = np.asarray(pp)[o]
    yy = np.asarray(yy)[o]
    return ts, rr, pp, yy

def read_float32(bag_dir, topic):
    so = StorageOptions(uri=bag_dir, storage_id=detect_storage_id(bag_dir))
    co = ConverterOptions(input_serialization_format="", output_serialization_format="")
    r = SequentialReader(); r.open(so, co)
    type_map = get_type_map(r)
    if topic not in type_map:
        raise RuntimeError(f"Topic '{topic}' not found. Available: {list(type_map.keys())}")
    ros_type = type_map[topic]
    MsgType = get_message(ros_type)
    if not ros_type.endswith("std_msgs/msg/Float32"):
        raise RuntimeError(f"Topic '{topic}' is type '{ros_type}', expected std_msgs/msg/Float32.")
    ts, vs = [], []
    while r.has_next():
        try:
            topic_name, data, t = r.read_next()
        except TypeError:
            t, data, topic_name = r.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(data, MsgType)
        ts.append(t * 1e-9)
        vs.append(float(msg.data))
    if not ts:
        raise RuntimeError(f"No samples on topic '{topic}'.")
    o = np.argsort(ts)
    ts = np.asarray(ts)[o]
    vs = np.asarray(vs)[o]
    return ts, vs


# ---------------- metrics helpers ----------------
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def overlapped_range(t1, t2, skip_sec=0.0):
    tmin = max(t1[0], t2[0]) + float(skip_sec)
    tmax = min(t1[-1], t2[-1])
    return (tmin, tmax) if tmax > tmin else (None, None)

def resample(t_ref, v_ref, t_act, v_act, skip_sec=0.0, n=3000):
    tmin, tmax = overlapped_range(t_ref, t_act, skip_sec)
    if tmin is None:
        return None, None, None
    tt = np.linspace(tmin, tmax, n)
    r = np.interp(tt, t_ref, v_ref)
    a = np.interp(tt, t_act, v_act)
    return tt, r, a

def estimate_lag(tt, r, a):
    r0 = r - np.mean(r)
    a0 = a - np.mean(a)
    corr = np.correlate(a0, r0, mode="full")   # shift a relative to r
    lag_idx = corr.argmax() - (len(a0) - 1)
    dt = lag_idx * (tt[1] - tt[0])
    return dt

def set_equal_3d(ax, x, y, z):
    xr = np.max(x) - np.min(x)
    yr = np.max(y) - np.min(y)
    zr = np.max(z) - np.min(z)
    r = max(xr, yr, zr, 1.0)
    cx = 0.5 * (np.max(x) + np.min(x))
    cy = 0.5 * (np.max(y) + np.min(y))
    cz = 0.5 * (np.max(z) + np.min(z))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(cx - r / 2, cx + r / 2)
    ax.set_ylim(cy - r / 2, cy + r / 2)
    ax.set_zlim(cz - r / 2, cz + r / 2)

def find_saturation_segments(t, u_mat, u_min=MIN_THRUST, u_max=MAX_THRUST, eps=SAT_EPS):
    """
    t      : (N,) time array
    u_mat  : (N, M) actuator matrix, one column per motor
    returns list of (t_start, t_end) where ANY motor is saturated
    """
    if t.size == 0:
        return []
    sat_any = ((u_mat <= u_min + eps) | (u_mat >= u_max - eps)).any(axis=1)
    if not np.any(sat_any):
        return []

    sat = sat_any.astype(int)
    changes = np.diff(sat)

    starts = []
    ends = []

    # start where it goes 0 -> 1
    starts_idx = np.where(changes == 1)[0] + 1
    # end where it goes 1 -> 0
    ends_idx = np.where(changes == -1)[0] + 1

    # handle if saturation already active at first sample
    if sat[0] == 1:
        starts_idx = np.r_[0, starts_idx]
    # handle if still active at last sample
    if sat[-1] == 1:
        ends_idx = np.r_[ends_idx, len(sat) - 1]

    for i0, i1 in zip(starts_idx, ends_idx):
        if i1 > i0:
            starts.append(t[i0])
            ends.append(t[i1])

    return list(zip(starts, ends))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot XY + 3D + RPY (with lag- and skip-aware RMS) from a rosbag2 directory."
    )
    ap.add_argument("bag")
    ap.add_argument("--ref", default="/debug/ref_xyz")
    ap.add_argument("--act", default="/debug/xyz")
    ap.add_argument("--rpy-ref", default="/debug/rpy_ref")
    ap.add_argument("--rpy-act", default="/debug/rpy")
    ap.add_argument("--skip-sec", type=float, default=0.0,
                    help="ignore first N seconds when computing errors")
    ap.add_argument("--save", default=None)
    ap.add_argument("--save3d", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.bag) or not os.path.exists(os.path.join(args.bag, "metadata.yaml")):
        print("ERROR: not a rosbag2 directory:", args.bag)
        sys.exit(2)

    # --- XYZ (positions) ---
    t_r, xr, yr, zr = read_xyz(args.bag, args.ref)
    t_a, xa, ya, za = read_xyz(args.bag, args.act)

    # --- Optional RPY and actuator reads (just data collection for now) ---
    has_rpy = False
    has_act = False

    try:
        tr_ref, roll_ref, pitch_ref, yaw_ref = read_rpy(args.bag, args.rpy_ref)
        tr_act, roll_act, pitch_act, yaw_act = read_rpy(args.bag, args.rpy_act)
        has_rpy = True
    except RuntimeError as e:
        print("RPY read skipped:", e)

    try:
        t_p1, u1 = read_float32(args.bag, "/debug/prop1")
        t_p2, u2 = read_float32(args.bag, "/debug/prop2")
        t_p3, u3 = read_float32(args.bag, "/debug/prop3")
        t_p4, u4 = read_float32(args.bag, "/debug/prop4")
        t_dc, dc = read_float32(args.bag, "/debug/dir_cos")
        has_act = True
    except RuntimeError as e:
        print("Actuator/dir_cos read skipped:", e)

    # --- Global time normalization: t = 0 at earliest available message ---
    all_starts = [t_r[0], t_a[0]]
    if has_rpy:
        all_starts += [tr_ref[0], tr_act[0]]
    if has_act:
        all_starts += [t_p1[0], t_p2[0], t_p3[0], t_p4[0], t_dc[0]]

    t0 = min(all_starts)
    print(f"Normalizing time: t=0 at approx. bag start (t0 = {t0:.3f} s in ROS time).")

    # shift all times to start at t=0
    t_r = t_r - t0
    t_a = t_a - t0
    if has_rpy:
        tr_ref = tr_ref - t0
        tr_act = tr_act - t0
    if has_act:
        t_p1 = t_p1 - t0
        t_p2 = t_p2 - t0
        t_p3 = t_p3 - t0
        t_p4 = t_p4 - t0
        t_dc = t_dc - t0

    # estimate global end time and set plotting max = min(50, T_end)
    t_end_candidates = [t_r[-1], t_a[-1]]
    if has_rpy:
        t_end_candidates += [tr_ref[-1], tr_act[-1]]
    if has_act:
        t_end_candidates += [t_p1[-1], t_p2[-1], t_p3[-1], t_p4[-1], t_dc[-1]]
    t_end = max(t_end_candidates)
    tmax_plot = min(100.0, t_end)

    # --- XY overlay (no time axis) ---
    plt.figure()
    plt.plot(xr, yr, "--", label="reference")
    plt.plot(xa, ya, "-", label="actual")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("XY ground track: reference vs actual")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=180)
        print("Saved:", args.save)

    # --- RMS XY (plain + lag-corrected), after skip ---
        # --- RMS position + error plots (XY + Z), after skip ---
    out = resample(t_r, xr, t_a, xa, skip_sec=args.skip_sec)
    if out is not None:
        tt, xr_i, xa_i = out
        _, yr_i, ya_i = resample(t_r, yr, t_a, ya, skip_sec=args.skip_sec)
        _, zr_i, za_i = resample(t_r, zr, t_a, za, skip_sec=args.skip_sec)

        # errori posizione
        ex = xa_i - xr_i
        ey = ya_i - yr_i
        ez = za_i - zr_i          # usa ref z dal topic
        # se la quota di riferimento è SEMPRE 5 m, usa invece:
        # ez = za_i - 5.0

        # RMS (senza correzione di lag)
        rms_xy  = np.sqrt(np.mean(ex**2 + ey**2))
        rms_z   = np.sqrt(np.mean(ez**2))
        rms_xyz = np.sqrt(np.mean(ex**2 + ey**2 + ez**2))
        print(f"RMS XY   (skip {args.skip_sec:.1f}s, no lag corr): {rms_xy:.3f} m")
        print(f"RMS Z    (skip {args.skip_sec:.1f}s, no lag corr): {rms_z:.3f} m")
        print(f"RMS XYZ  (skip {args.skip_sec:.1f}s, no lag corr): {rms_xyz:.3f} m")

        # stima lag XY e RMS corretti
        dt_x = estimate_lag(tt, xr_i, xa_i)
        dt_y = estimate_lag(tt, yr_i, ya_i)
        dt = 0.5 * (dt_x + dt_y)

        xr_s = np.interp(tt, t_r + dt, xr)
        yr_s = np.interp(tt, t_r + dt, yr)
        zr_s = np.interp(tt, t_r + dt, zr)

        ex2 = xa_i - xr_s
        ey2 = ya_i - yr_s
        ez2 = za_i - zr_s       # o za_i - 5.0 se usi quota costante

        rms_xy_lag  = np.sqrt(np.mean(ex2**2 + ey2**2))
        rms_xyz_lag = np.sqrt(np.mean(ex2**2 + ey2**2 + ez2**2))

        print(f"Estimated lag dt_x={dt_x:.3f}s dt_y={dt_y:.3f}s → avg dt={dt:.3f}s")
        print(f"RMS XY   (skip {args.skip_sec:.1f}s, lag-corr): {rms_xy_lag:.3f} m")
        print(f"RMS XYZ  (skip {args.skip_sec:.1f}s, lag-corr): {rms_xyz_lag:.3f} m")

        # --- NUOVO: plot degli errori nel tempo ---
        fig_e, axe = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

        axe[0].plot(tt, ex, label="e_x = x - x_ref")
        axe[0].axhline(0.0, linestyle="--", linewidth=0.8)
        axe[0].set_ylabel("e_x [m]")
        axe[0].grid(True)
        axe[0].legend(loc="best")

        axe[1].plot(tt, ey, label="e_y = y - y_ref")
        axe[1].axhline(0.0, linestyle="--", linewidth=0.8)
        axe[1].set_ylabel("e_y [m]")
        axe[1].grid(True)
        axe[1].legend(loc="best")

        axe[2].plot(tt, ez, label="e_z = z - z_ref")
        axe[2].axhline(0.0, linestyle="--", linewidth=0.8)
        axe[2].set_ylabel("e_z [m]")
        axe[2].set_xlabel("t [s]")
        axe[2].grid(True)
        axe[2].legend(loc="best")

        axe[2].set_xlim(0.0, tmax_plot)
        fig_e.suptitle("Position tracking error vs time")
        fig_e.tight_layout(rect=[0, 0.03, 1, 0.97])


        # time plots
        plt.figure(figsize=(12, 5))
        plt.plot(tt, xr_i, "--", label="x_ref")
        plt.plot(tt, xa_i, "-", label="x")
        plt.plot(tt, yr_i, "--", label="y_ref")
        plt.plot(tt, ya_i, "-", label="y")
        plt.grid(True)
        plt.xlabel("t [s]")
        plt.ylabel("position")
        plt.title("Position vs reference")
        plt.legend()
        plt.xlim(0.0, tmax_plot)
        plt.tight_layout()

    # --- 3D trajectory (z inverted PX4 -> Z up) ---
    if HAS_3D:
        zr_plot = -zr
        za_plot = -za

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(xr, yr, zr_plot, "--", label="reference")
        ax.plot(xa, ya, za_plot, "-", label="actual")

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m, up]")
        ax.set_title("3D trajectory: reference vs actual (PX4 z inverted)")

        set_equal_3d(
            ax,
            np.r_[xr, xa],
            np.r_[yr, ya],
            np.r_[zr_plot, za_plot],
        )

        # z-axis range constraint as you had
        ax.set_zlim(-1.0, 8.0)

        ax.legend()
        if args.save3d:
            plt.savefig(args.save3d, dpi=180)
            print("Saved 3D:", args.save3d)
    else:
        print("3D backend not available; skipping 3D plot.")

    # --- RPY with RMS (if available) ---
    if has_rpy:
        out_r = resample(tr_ref, roll_ref, tr_act, roll_act,
                         skip_sec=args.skip_sec, n=4000)
        out_p = resample(tr_ref, pitch_ref, tr_act, pitch_act,
                         skip_sec=args.skip_sec, n=4000)
        out_y = resample(tr_ref, yaw_ref, tr_act, yaw_act,
                         skip_sec=args.skip_sec, n=4000)
        if out_r and out_p and out_y:
            ttr, r_ref_i, r_act_i = out_r
            _, p_ref_i, p_act_i = out_p
            _, y_ref_i, y_act_i = out_y

            er = r_act_i - r_ref_i
            ep = p_act_i - p_ref_i
            eyaw = (y_act_i - y_ref_i + np.pi) % (2 * np.pi) - np.pi
            rms_r = np.sqrt(np.mean(er ** 2))
            rms_p = np.sqrt(np.mean(ep ** 2))
            rms_y = np.sqrt(np.mean(eyaw ** 2))
            print(f"RMS roll  [rad]: {rms_r:.4f}  [{np.degrees(rms_r):.2f} deg]")
            print(f"RMS pitch [rad]: {rms_p:.4f}  [{np.degrees(rms_p):.2f} deg]")
            print(f"RMS yaw   [rad]: {rms_y:.4f}  [{np.degrees(rms_y):.2f} deg]")

            fig2, axr = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
            axr[0].plot(ttr, r_ref_i, "--", label="roll_ref")
            axr[0].plot(ttr, r_act_i, "-", label="roll")
            axr[1].plot(ttr, p_ref_i, "--", label="pitch_ref")
            axr[1].plot(ttr, p_act_i, "-", label="pitch")
            axr[2].plot(ttr, y_ref_i, "--", label="yaw_ref")
            axr[2].plot(ttr, y_act_i, "-", label="yaw")
            for a in axr:
                a.grid(True)
            axr[0].set_ylabel("roll [rad]")
            axr[1].set_ylabel("pitch [rad]")
            axr[2].set_ylabel("yaw [rad]")
            axr[2].set_xlabel("t [s]")
            axr[0].legend()
            axr[1].legend()
            axr[2].legend()
            axr[2].set_xlim(0.0, tmax_plot)
            fig2.suptitle("Attitude vs reference")
            fig2.tight_layout(rect=[0, 0.03, 1, 0.97])

    # --- Actuator commands + directional cosine (if available) ---
    if has_act:
        # build common time window across all 5 signals
        t_starts = [t_p1[0], t_p2[0], t_p3[0], t_p4[0], t_dc[0]]
        t_ends = [t_p1[-1], t_p2[-1], t_p3[-1], t_p4[-1], t_dc[-1]]
        tmin = max(t_starts)
        tmax = min(t_ends)
        if tmax <= tmin:
            print("No common time interval between motors and dir_cos.")
        else:
            # common high-resolution time grid
            N_grid = 2500
            tt = np.linspace(tmin, tmax, N_grid)

            u1_i = np.interp(tt, t_p1, u1)
            u2_i = np.interp(tt, t_p2, u2)
            u3_i = np.interp(tt, t_p3, u3)
            u4_i = np.interp(tt, t_p4, u4)
            dc_i = np.interp(tt, t_dc, dc)

            U = np.vstack((u1_i, u2_i, u3_i, u4_i)).T     # shape (N_grid, 4)
            sat_segments = find_saturation_segments(tt, U)
                        # ---------- quantitative metrics for dir_cos ----------
            # basic stats
            dc_mean = np.mean(dc_i)
            dc_std  = np.std(dc_i)
            dc_min  = np.min(dc_i)
            dc_max  = np.max(dc_i)

            p1, p5, p50, p95, p99 = np.percentile(dc_i, [1, 5, 50, 95, 99])

            # thresholds
            frac_ge_099  = np.mean(dc_i >= 0.99)
            frac_ge_0995 = np.mean(dc_i >= 0.995)

            # approximate misalignment angle statistics
            dc_clip = np.clip(dc_i, -1.0, 1.0)
            theta   = np.arccos(dc_clip)           # [rad]
            theta_rms = np.sqrt(np.mean(theta**2))
            theta_max = np.max(theta)

            print("\nDirectional cosine metrics over actuator interval:")
            print(f"  dir_cos mean      = {dc_mean:.4f}")
            print(f"  dir_cos std       = {dc_std:.4f}")
            print(f"  dir_cos min, max  = {dc_min:.4f}, {dc_max:.4f}")
            print(f"  percentiles [1,5,50,95,99] = "
                  f"{p1:.4f}, {p5:.4f}, {p50:.4f}, {p95:.4f}, {p99:.4f}")
            print(f"  time frac dir_cos >= 0.99  = {100.0*frac_ge_099:5.1f}%")
            print(f"  time frac dir_cos >= 0.995 = {100.0*frac_ge_0995:5.1f}%")
            print(f"  RMS misalignment angle = {np.degrees(theta_rms):.3f} deg")
            print(f"  max misalignment angle = {np.degrees(theta_max):.3f} deg")


            fig_act, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # top: actuator commands
            ax[0].plot(tt, u1_i, label="u1")
            ax[0].plot(tt, u2_i, label="u2")
            ax[0].plot(tt, u3_i, label="u3")
            ax[0].plot(tt, u4_i, label="u4")
            ax[0].axhline(MIN_THRUST, linestyle="--", linewidth=0.8, label="u_min")
            ax[0].axhline(MAX_THRUST, linestyle="--", linewidth=0.8, label="u_max")
            ax[0].set_ylabel("motor command u_i [-]")
            ax[0].grid(True)
            ax[0].legend(loc="best")
            ax[0].set_title("Actuator commands and directional cosine")

            # bottom: dir_cos
            ax[1].plot(tt, dc_i, label="dir_cos", linewidth=2.0)
            ax[1].axhline(1.0, linestyle="--", linewidth=1.0, label="ideal = 1")
            ax[1].set_ylim(0.0, 1.05)
            ax[1].set_xlabel("t [s]")
            ax[1].set_ylabel("dir_cos [-]")
            ax[1].grid(True)
            ax[1].legend(loc="best")

            # x-axis limit 0..min(50, T_end)
            ax[1].set_xlim(0.0, tmax_plot)

                        # build a boolean mask for saturation, to reuse for metrics and plot
            sat_mask = np.zeros_like(tt, dtype=bool)
            for t0_seg, t1_seg in sat_segments:
                sat_mask |= (tt >= t0_seg) & (tt <= t1_seg)

            # highlight saturation regions on both panels
            for (t0_seg, t1_seg) in sat_segments:
                for a in ax:
                    a.axvspan(t0_seg, t1_seg, alpha=0.15)

            # additionally emphasize dir_cos during saturation
            if sat_segments:
                ax[1].scatter(tt[sat_mask], dc_i[sat_mask], s=10)

                # metrics in and out of saturation
                dc_sat    = dc_i[sat_mask]
                dc_nosat  = dc_i[~sat_mask]
                if dc_sat.size > 0:
                    dc_sat_mean   = np.mean(dc_sat)
                    dc_nosat_mean = np.mean(dc_nosat) if dc_nosat.size > 0 else np.nan

                    theta_sat   = np.arccos(np.clip(dc_sat, -1.0, 1.0))
                    theta_nosat = np.arccos(np.clip(dc_nosat, -1.0, 1.0)) if dc_nosat.size > 0 else None
                    theta_sat_rms   = np.sqrt(np.mean(theta_sat**2))
                    theta_nosat_rms = np.sqrt(np.mean(theta_nosat**2)) if theta_nosat is not None else np.nan

                    frac_time_sat = np.mean(sat_mask)

                    print("\nDirectional cosine vs saturation:")
                    print(f"  time in saturation           = {100.0*frac_time_sat:5.1f}%")
                    print(f"  mean dir_cos in saturation   = {dc_sat_mean:.4f}")
                    print(f"  mean dir_cos outside sat     = {dc_nosat_mean:.4f}")
                    print(f"  RMS angle in saturation      = {np.degrees(theta_sat_rms):.3f} deg")
                    if theta_nosat is not None:
                        print(f"  RMS angle outside saturation = {np.degrees(theta_nosat_rms):.3f} deg")

            fig_act.tight_layout()


    plt.show()


if __name__ == "__main__":
    main()
