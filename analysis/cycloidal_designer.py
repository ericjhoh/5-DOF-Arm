"""
Cycloidal reducer designer / checker  ->  SolidWorks global variables

Change the disc parameters with the sliders, watch the disc mesh as you turn the
input, read the design checks, then click "Export" to write:

  exports/<name>_Global_Variables.txt   SolidWorks Equations > Import (same format as
                                        your 30_1_Global_Variables.txt, plus CLR, T_DISK
                                        and derived equations that update in SolidWorks)
  exports/<name>_profile_xyz.txt        disc profile points (mm) for
                                        Insert > Curve > Curve Through XYZ Points
  exports/<name>_equation_curve.txt     X(t), Y(t) for an Equation Driven Curve,
                                        written in terms of the global variables

Variable names match your existing files:
  R_ROLL      ring roller radius                 D_SUN       ring pin circle diameter
  NUM_ROLL    number of ring rollers (N)         RHO         eccentricity
  D_SHAFT     disc center bore (cam bearing OD)  D_OUTPUT    output pin circle diameter
  NUM_OUT     number of output rollers           R_OUT_ROLL  output roller radius
  CLR         print clearance (profile offset + output hole clearance)
  T_DISK      disc thickness

Run:  python cycloidal_designer.py          (needs numpy + matplotlib)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.patches import Circle

PRESETS = {
    "30:1": dict(R_ROLL=5, D_SUN=120, NUM_ROLL=31, RHO=1.0, D_SHAFT=28, D_OUTPUT=80,
                 NUM_OUT=8, R_OUT_ROLL=5, CLR=0.10, T_DISK=8, TORQUE=7.65),
    "15:1": dict(R_ROLL=5, D_SUN=80, NUM_ROLL=16, RHO=1.0, D_SHAFT=15, D_OUTPUT=40,
                 NUM_OUT=6, R_OUT_ROLL=5, CLR=0.10, T_DISK=8, TORQUE=4.14),
}
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
NOZZLE = 0.4  # mm, used for the minimum-wall check


# --------------------------------------------------------------------- geometry
def profile(p, n_pts=6000):
    """Disc profile (mm) about the disc center, using the same equation as your
    gearbox parameter files, with the roller radius grown by CLR (equidistant offset)."""
    R, E, N = p["D_SUN"] / 2, p["RHO"], int(p["NUM_ROLL"])
    Rr = p["R_ROLL"] + p["CLR"]
    t = np.linspace(0, 2 * np.pi, n_pts)
    psi = np.arctan(np.sin((1 - N) * t) / ((R / (E * N)) - np.cos((1 - N) * t)))
    x = R * np.cos(t) - Rr * np.cos(t + psi) - E * np.cos(N * t)
    y = -R * np.sin(t) + Rr * np.sin(t + psi) + E * np.sin(N * t)
    # pitch curve (path of a roller center), used for the undercut check
    xp = R * np.cos(t) - E * np.cos(N * t)
    yp = -R * np.sin(t) + E * np.sin(N * t)
    return t, x, y, xp, yp


def checks(p):
    R, E, N = p["D_SUN"] / 2, p["RHO"], int(p["NUM_ROLL"])
    Rr, clr = p["R_ROLL"], p["CLR"]
    n_out, r_out = int(p["NUM_OUT"]), p["D_OUTPUT"] / 2
    hole_d = 2 * p["R_OUT_ROLL"] + 2 * E + 2 * clr
    t, x, y, xp, yp = profile(p)

    # signed curvature of the pitch curve; convex side = sign of the average curvature
    dx, dy = np.gradient(xp, t), np.gradient(yp, t)
    ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
    k = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
    orient = np.sign(np.sum(xp[:-1] * yp[1:] - xp[1:] * yp[:-1]))  # curve direction (shoelace)
    convex = np.sign(k) == orient
    rho_min = (1 / np.abs(k[convex])).min()
    # tangent reversal between pitch curve and profile = loop / undercut
    px, py = np.gradient(x, t), np.gradient(y, t)
    loops = np.any(dx * px + dy * py <= 0)

    r_prof = np.hypot(x, y)
    root = r_prof.min()
    out = {
        "RATIO": N - 1,
        "K1": E * N / R,
        "rho_min": rho_min,
        "undercut_margin": rho_min - (Rr + clr),
        "loops": loops,
        "tooth_depth": r_prof.max() - root,
        "root": root,
        "roller_gap": 2 * R * np.sin(np.pi / N) - 2 * Rr,
        "hole_d": hole_d,
        "wall_outer": root - (r_out + hole_d / 2),
        "wall_between": 2 * r_out * np.sin(np.pi / n_out) - hole_d,
        "wall_inner": (r_out - hole_d / 2) - p["D_SHAFT"] / 2,
        "bl_profile": np.degrees(clr / (E * N)),
        "bl_output": np.degrees(clr / r_out),
        "cam_load": p["TORQUE"] / ((N - 1) * E / 1000) / 2,  # N per disc, order of magnitude
    }
    out["bl_total"] = out["bl_profile"] + out["bl_output"]
    return out, (x, y)


def grade(ok, warn):
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def report(c):
    min_wall = 4 * NOZZLE
    rows = [
        ("Ratio", f"{c['RATIO']}:1", "PASS"),
        ("K1 = e·N/R (0.5–0.8)", f"{c['K1']:.2f}",
         grade(0.5 <= c["K1"] <= 0.8, 0.4 <= c["K1"] < 0.5 or 0.8 < c["K1"] <= 0.9)),
        ("Undercut margin ρmin − (Rr+CLR)", f"{c['undercut_margin']:.2f} mm",
         "FAIL" if c["loops"] else grade(c["undercut_margin"] > 1, c["undercut_margin"] > 0)),
        ("Tooth depth", f"{c['tooth_depth']:.2f} mm", grade(c["tooth_depth"] >= 1.95, c["tooth_depth"] >= 1.45)),
        ("Gap between ring rollers", f"{c['roller_gap']:.2f} mm", grade(c["roller_gap"] >= 1, c["roller_gap"] > 0.3)),
        ("Output hole Ø (2r+2e+2CLR)", f"{c['hole_d']:.2f} mm", "PASS"),
        ("Wall: hole → tooth root", f"{c['wall_outer']:.2f} mm", grade(c["wall_outer"] >= 2 * min_wall, c["wall_outer"] >= min_wall)),
        ("Wall: between holes", f"{c['wall_between']:.2f} mm", grade(c["wall_between"] >= 2 * min_wall, c["wall_between"] >= min_wall)),
        ("Wall: hole → center bore", f"{c['wall_inner']:.2f} mm", grade(c["wall_inner"] >= 2 * min_wall, c["wall_inner"] >= min_wall)),
        ("Backlash from CLR (profile)", f"{c['bl_profile']:.3f}° ({c['bl_profile']*60:.1f}′)", "INFO"),
        ("Backlash from CLR (output)", f"{c['bl_output']:.3f}° ({c['bl_output']*60:.1f}′)", "INFO"),
        ("Backlash total (est.)", f"{c['bl_total']:.3f}° ({c['bl_total']*60:.1f}′)", "INFO"),
        ("Cam bearing load / disc (est.)", f"{c['cam_load']:.0f} N", "INFO"),
    ]
    return rows


# ---------------------------------------------------------------------- export
def sw_val(v, unit="mm"):
    return f"{v:g}{unit}"


def export(p, name):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    base = os.path.join(EXPORT_DIR, name)
    lines = [
        f'"R_ROLL"= {sw_val(p["R_ROLL"])}',
        f'"D_SUN"= {sw_val(p["D_SUN"])}',
        f'"NUM_ROLL"= {int(p["NUM_ROLL"])}',
        f'"RHO"= {sw_val(p["RHO"])}',
        f'"D_SHAFT"= {sw_val(p["D_SHAFT"])}',
        f'"D_OUTPUT"= {sw_val(p["D_OUTPUT"])}',
        f'"NUM_OUT"= {int(p["NUM_OUT"])}',
        f'"R_OUT_ROLL"= {sw_val(p["R_OUT_ROLL"])}',
        f'"CLR"= {sw_val(p["CLR"])}',
        f'"T_DISK"= {sw_val(p["T_DISK"])}',
        # derived: these stay live in SolidWorks
        '"RATIO"= "NUM_ROLL" - 1',
        '"D_OUT_HOLE"= 2 * "R_OUT_ROLL" + 2 * "RHO" + 2 * "CLR"',
        '"K1"= "RHO" * "NUM_ROLL" / ( "D_SUN" / 2 )',
    ]
    # SolidWorks' own export: UTF-8 with BOM, CRLF
    with open(base + "_Global_Variables.txt", "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")

    _, x, y, _, _ = profile(p, n_pts=int(p["NUM_ROLL"]) * 60 + 1)
    with open(base + "_profile_xyz.txt", "w", newline="\r\n") as f:
        for xi, yi in zip(x[:-1], y[:-1]):  # last point repeats the first; use 'Closed curve'
            f.write(f"{xi:.5f}\t{yi:.5f}\t0\n")

    R, Rr, E, N = '("D_SUN"/2)', '("R_ROLL"+"CLR")', '"RHO"', '"NUM_ROLL"'
    psi = f"atn(sin((1-{N})*t)/(({R}/({E}*{N}))-cos((1-{N})*t)))"
    xe = f"({R}*cos(t))-({Rr}*cos(t+{psi}))-({E}*cos({N}*t))"
    ye = f"(-{R}*sin(t))+({Rr}*sin(t+{psi}))+({E}*sin({N}*t))"
    xn = (f"({p['D_SUN']/2:g}*cos(t))-({p['R_ROLL']+p['CLR']:g}*cos(t+atn(sin((1-{int(p['NUM_ROLL'])})*t)/"
          f"(({p['D_SUN']/2:g}/({p['RHO']:g}*{int(p['NUM_ROLL'])}))-cos((1-{int(p['NUM_ROLL'])})*t)))))"
          f"-({p['RHO']:g}*cos({int(p['NUM_ROLL'])}*t))")
    yn = (f"(-{p['D_SUN']/2:g}*sin(t))+({p['R_ROLL']+p['CLR']:g}*sin(t+atn(sin((1-{int(p['NUM_ROLL'])})*t)/"
          f"(({p['D_SUN']/2:g}/({p['RHO']:g}*{int(p['NUM_ROLL'])}))-cos((1-{int(p['NUM_ROLL'])})*t)))))"
          f"+({p['RHO']:g}*sin({int(p['NUM_ROLL'])}*t))")
    with open(base + "_equation_curve.txt", "w", newline="\r\n") as f:
        f.write("Equation Driven Curve (Parametric), t1 = 0, t2 = 2*pi\n\n")
        f.write("Linked to global variables (use if your SolidWorks version accepts them in the curve):\n")
        f.write(f"Xt = {xe}\nYt = {ye}\n\n")
        f.write("Numeric (always works; re-export after changing parameters):\n")
        f.write(f"Xt = {xn}\nYt = {yn}\n")
    return base


# ------------------------------------------------------------------------- GUI
fig = plt.figure(figsize=(14, 8.5))
fig.canvas.manager.set_window_title("Cycloidal reducer designer")
ax = fig.add_axes([0.03, 0.30, 0.50, 0.66])
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
tax = fig.add_axes([0.56, 0.30, 0.42, 0.66]); tax.axis("off")

slider_specs = [  # key, label, min, max, step
    ("R_ROLL", "R_ROLL (mm)", 1.5, 10, 0.5),
    ("D_SUN", "D_SUN (mm)", 40, 200, 1),
    ("NUM_ROLL", "NUM_ROLL", 8, 61, 1),
    ("RHO", "RHO e (mm)", 0.3, 4, 0.05),
    ("D_SHAFT", "D_SHAFT (mm)", 5, 60, 0.5),
    ("D_OUTPUT", "D_OUTPUT (mm)", 15, 160, 1),
    ("NUM_OUT", "NUM_OUT", 3, 14, 1),
    ("R_OUT_ROLL", "R_OUT_ROLL (mm)", 1.5, 10, 0.5),
    ("CLR", "CLR (mm)", 0, 0.4, 0.01),
    ("T_DISK", "T_DISK (mm)", 3, 20, 0.5),
    ("TORQUE", "output torque (N·m)", 0.5, 30, 0.05),
]
sliders = {}
p0 = PRESETS["30:1"]
for i, (key, label, lo, hi, st) in enumerate(slider_specs):
    col, row = divmod(i, 6)
    sax = fig.add_axes([0.10 + col * 0.45, 0.235 - row * 0.032, 0.30, 0.02])
    sliders[key] = Slider(sax, label, lo, hi, valinit=p0[key], valstep=st)
th_ax = fig.add_axes([0.55, 0.075, 0.30, 0.02])
s_theta = Slider(th_ax, "input angle (°)", 0, 360 * 3, valinit=0, valstep=1)

name_ax = fig.add_axes([0.10, 0.02, 0.16, 0.035])
name_box = TextBox(name_ax, "name ", initial="30_1")
b30 = Button(fig.add_axes([0.30, 0.02, 0.07, 0.035]), "Preset 30:1")
b15 = Button(fig.add_axes([0.38, 0.02, 0.07, 0.035]), "Preset 15:1")
bexp = Button(fig.add_axes([0.47, 0.02, 0.09, 0.035]), "Export")
status = fig.text(0.58, 0.03, "", fontsize=9)

COLORS = {"PASS": "#2E7D4F", "WARN": "#B07A00", "FAIL": "#B3261E", "INFO": "#444444"}


def params():
    return {k: s.val for k, s in sliders.items()}


def draw(_=None):
    p = params()
    R, E, N = p["D_SUN"] / 2, p["RHO"], int(p["NUM_ROLL"])
    th = np.radians(s_theta.val)
    phi = -th / (N - 1)
    c, info = checks(p)
    x, y = info
    ax.clear(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    lim = R + p["R_ROLL"] + 6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    def place(px, py, ang, cx, cy):
        ca, sa = np.cos(ang), np.sin(ang)
        return ca * px - sa * py + cx, sa * px + ca * py + cy

    # second disc, 180° out of phase (faint)
    x2, y2 = place(x, y, -(th + np.pi) / (N - 1) + 0, E * np.cos(th + np.pi), E * np.sin(th + np.pi))
    ax.plot(x2, y2, color="0.75", lw=0.8, ls="--")
    # main disc
    X, Y = place(x, y, phi, E * np.cos(th), E * np.sin(th))
    ax.fill(X, Y, color="#E8743B", alpha=0.25)
    ax.plot(X, Y, color="#B04A1A", lw=1.2)
    cx, cy = E * np.cos(th), E * np.sin(th)
    ax.add_patch(Circle((cx, cy), p["D_SHAFT"] / 2, fill=False, color="#B04A1A", lw=1))
    # ring rollers (fixed)
    for k in range(N):
        a = 2 * np.pi * k / N
        ax.add_patch(Circle((R * np.cos(a), R * np.sin(a)), p["R_ROLL"], color="#2F6FB0", alpha=0.85))
    # output holes (move with disc) and output rollers (on output flange, rotate by phi)
    r_out = p["D_OUTPUT"] / 2
    for k in range(int(p["NUM_OUT"])):
        a = 2 * np.pi * k / int(p["NUM_OUT"]) + phi
        ax.add_patch(Circle((cx + r_out * np.cos(a), cy + r_out * np.sin(a)), c["hole_d"] / 2,
                            fill=False, color="#B04A1A", lw=1))
        ax.add_patch(Circle((r_out * np.cos(a), r_out * np.sin(a)), p["R_OUT_ROLL"], color="#15191F", alpha=0.8))
    ax.plot([0], [0], "k+", ms=10)
    ax.set_title(f"{c['RATIO']}:1 · input {s_theta.val:.0f}° → output {np.degrees(phi):.2f}°   "
                 "(dashed = second disc)", fontsize=10)

    tax.clear(); tax.axis("off")
    tax.text(0, 1.0, "Design checks", fontsize=13, weight="bold", va="top")
    for i, (label, val, g) in enumerate(report(c)):
        yy = 0.93 - i * 0.062
        tax.text(0.00, yy, label, fontsize=10, va="top")
        tax.text(0.62, yy, val, fontsize=10, va="top", family="monospace")
        tax.text(0.90, yy, g, fontsize=10, va="top", weight="bold", color=COLORS[g])
    tax.text(0, 0.06, f"Walls graded against {4*NOZZLE:.1f} mm (4 × {NOZZLE} mm nozzle). Backlash and cam load are\n"
             "first-order estimates; measure real backlash with a dial indicator.",
             fontsize=8.5, color="0.35", va="top")
    fig.canvas.draw_idle()


def set_preset(name):
    for k, v in PRESETS[name].items():
        sliders[k].set_val(v)
    name_box.set_val(name.replace(":", "_"))


def do_export(_):
    name = name_box.text.strip() or "cycloidal"
    base = export(params(), name)
    status.set_text(f"Exported to {os.path.dirname(base)}\\{name}_*.txt")
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(draw)
s_theta.on_changed(draw)
b30.on_clicked(lambda _: set_preset("30:1"))
b15.on_clicked(lambda _: set_preset("15:1"))
bexp.on_clicked(do_export)

if __name__ == "__main__":
    draw()
    plt.show()
