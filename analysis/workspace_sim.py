"""
5-DOF arm workspace simulator (side view of the J2/J3 pitch plane).

Shows every tool-tip position the arm can reach within its joint limits (J4 = 0),
the arc where the gripper points straight down, and the arm at a chosen pose.

Geometry (matches 5-DOF-Arm_v3.xlsx):
    L1  = J2 shoulder -> J3 elbow                     250 mm
    LF  = J3 -> J5 along the forearm (L2 + L3)        25 + 125 = 150 mm
    HP  = J5 -> tool tip, gripper hangs 90 deg off forearm   80 mm
    H   = shoulder (J2) axis height above the table   212 mm (from URDF)

Angles:
    shoulder = upper-arm angle from vertical (0 = straight up, + = leaning forward)
    elbow    = forearm angle relative to the upper arm
    The gripper points straight down when shoulder + elbow = 90 deg.

Run:  python workspace_sim.py        (needs numpy + matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---- default geometry (mm) and joint limits (deg) --------------------------
L1, LF, HP, H = 250.0, 150.0, 80.0, 212.0
SHOULDER_LIMIT = 125.0   # +/- from vertical; replace with your LeRobot calibration values
ELBOW_LIMIT = 135.0      # +/- relative to upper arm
START_POSE = (121.0, -31.0)   # forearm horizontal, gripper down, tip on the table


def forward_kinematics(a2, a3, l1, lf, hp, h):
    """Return (elbow, wrist, tip) xy points in mm for angles in degrees (arrays OK)."""
    a2r = np.radians(a2)
    ar = np.radians(np.asarray(a2) + np.asarray(a3))
    elbow = np.array([l1 * np.sin(a2r), h + l1 * np.cos(a2r)])
    wrist = elbow + np.array([lf * np.sin(ar), lf * np.cos(ar)])
    tip = wrist + np.array([hp * np.cos(ar), -hp * np.sin(ar)])  # gripper perpendicular to forearm
    return elbow, wrist, tip


def reachable_tips(l1, lf, hp, h, lim2, lim3, step=2.0):
    a2, a3 = np.meshgrid(np.arange(-lim2, lim2 + 1e-9, step), np.arange(-lim3, lim3 + 1e-9, step))
    _, _, tip = forward_kinematics(a2.ravel(), a3.ravel(), l1, lf, hp, h)
    keep = tip[1] >= 0
    return tip[:, keep]


def top_down_arc(l1, lf, hp, h, lim2, lim3):
    a2 = np.arange(-lim2, lim2 + 1e-9, 0.5)
    a3 = 90.0 - a2
    ok = np.abs(a3) <= lim3
    _, _, tip = forward_kinematics(a2[ok], a3[ok], l1, lf, hp, h)
    keep = tip[1] >= 0
    return tip[:, keep]


# ---- figure -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.40)
ax.set_aspect("equal")
ax.set_xlabel("distance from base axis (mm)")
ax.set_ylabel("height above table (mm)")
ax.set_title("5-DOF arm workspace — side view (J4 = 0)")
ax.grid(alpha=0.25)

cloud = ax.scatter([], [], s=3, c="0.55", alpha=0.45, label="reachable tool tip")
arc = ax.scatter([], [], s=9, c="#BA7517", label="gripper straight down")
outer = plt.Circle((0, H), L1 + LF, fill=False, ls="--", lw=0.8, color="0.4")
inner = plt.Circle((0, H), abs(L1 - LF), fill=False, ls="--", lw=0.8, color="0.4")
ax.add_patch(outer)
ax.add_patch(inner)
ax.axhline(0, color="0.2", lw=1.2)
(upper_line,) = ax.plot([], [], lw=7, color="#185FA5", solid_capstyle="round", label="upper arm L1")
(fore_line,) = ax.plot([], [], lw=5, color="#0F6E56", solid_capstyle="round", label="forearm L2+L3")
(grip_line,) = ax.plot([], [], lw=3, color="0.15", solid_capstyle="round", label="gripper")
(joints,) = ax.plot([], [], "o", ms=9, mfc="white", mec="0.15", mew=1.5)
(tip_pt,) = ax.plot([], [], "o", ms=7, color="#D85A30")
readout = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9,
                  bbox=dict(boxstyle="round", fc="white", ec="0.8"))
ax.legend(loc="upper right", fontsize=8)

# ---- sliders ----------------------------------------------------------------
def make_slider(row, label, vmin, vmax, vinit, step=1.0):
    sax = fig.add_axes([0.25, 0.29 - row * 0.035, 0.6, 0.022])
    return Slider(sax, label, vmin, vmax, valinit=vinit, valstep=step)

s_a2 = make_slider(0, "shoulder (° from vertical)", -150, 150, START_POSE[0])
s_a3 = make_slider(1, "elbow (° relative)", -150, 150, START_POSE[1])
s_l2 = make_slider(2, "shoulder limit ±°", 30, 150, SHOULDER_LIMIT)
s_l3 = make_slider(3, "elbow limit ±°", 30, 150, ELBOW_LIMIT)
s_L1 = make_slider(4, "L1 upper arm (mm)", 100, 400, L1, 5)
s_LF = make_slider(5, "forearm L2+L3 (mm)", 50, 300, LF, 5)
s_HP = make_slider(6, "gripper offset (mm)", 0, 150, HP, 5)
s_H = make_slider(7, "shoulder height (mm)", 50, 400, H, 5)

reset_ax = fig.add_axes([0.02, 0.02, 0.09, 0.04])
reset_btn = Button(reset_ax, "Reset")

_last_static = None


def update(_=None):
    global _last_static
    l1, lf, hp, h = s_L1.val, s_LF.val, s_HP.val, s_H.val
    lim2, lim3 = s_l2.val, s_l3.val
    a2 = float(np.clip(s_a2.val, -lim2, lim2))
    a3 = float(np.clip(s_a3.val, -lim3, lim3))

    static = (l1, lf, hp, h, lim2, lim3)
    if static != _last_static:  # only recompute the point cloud when geometry/limits change
        pts = reachable_tips(*static)
        cloud.set_offsets(pts.T if pts.size else np.empty((0, 2)))
        arc_pts = top_down_arc(*static)
        arc.set_offsets(arc_pts.T if arc_pts.size else np.empty((0, 2)))
        outer.set_center((0, h)); outer.set_radius(l1 + lf)
        inner.set_center((0, h)); inner.set_radius(abs(l1 - lf))
        span = l1 + lf + hp + 50
        ax.set_xlim(-span, span)
        ax.set_ylim(-30, h + l1 + lf + 50)
        _last_static = static

    elbow, wrist, tip = forward_kinematics(a2, a3, l1, lf, hp, h)
    upper_line.set_data([0, elbow[0]], [h, elbow[1]])
    fore_line.set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
    grip_line.set_data([wrist[0], tip[0]], [wrist[1], tip[1]])
    joints.set_data([0, elbow[0], wrist[0]], [h, elbow[1], wrist[1]])
    tip_pt.set_data([tip[0]], [tip[1]])

    tilt = (a2 + a3) - 90
    # where a vertical gripper meets the table (forearm horizontal)
    s = (hp - h) / l1
    table_ring = f"{l1 * np.sqrt(1 - s * s) + lf:5.0f} mm out" if abs(s) <= 1 else "not reachable"
    readout.set_text(
        f"tip: {tip[0]:6.0f} mm out, {tip[1]:6.0f} mm up{'  (below table!)' if tip[1] < 0 else ''}\n"
        f"gripper tilt from vertical: {tilt:+.0f}°\n"
        f"wrist ring: {abs(l1 - lf):.0f}–{l1 + lf:.0f} mm from shoulder\n"
        f"top-down grasp on table: {table_ring}"
    )
    fig.canvas.draw_idle()


for s in (s_a2, s_a3, s_l2, s_l3, s_L1, s_LF, s_HP, s_H):
    s.on_changed(update)
reset_btn.on_clicked(lambda _: [s.reset() for s in (s_a2, s_a3, s_l2, s_l3, s_L1, s_LF, s_HP, s_H)])

update()
plt.show()
