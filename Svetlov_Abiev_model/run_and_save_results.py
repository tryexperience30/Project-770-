"""Run the Svetlov–Abiev solver and write figures + tables to results/."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from svetlov_model import (
    Channel,
    critical_diameter,
    equilibrium_diameter,
    forces_on_droplet,
    mlh,
    real_diameter,
    solve_analytical,
    solve_chebyshev,
)

OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.name}")
    return path


def main() -> None:
    ch = Channel()
    R, mu1, mu2 = ch.R, ch.mu1, ch.mu2
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    log("Svetlov & Abiev (2021) solver results")
    log("=" * 60)
    log(f"Channel diameter dc = {ch.dc * 1e6:.1f} μm")
    log(f"Channel radius   R  = {R * 1e6:.1f} μm")
    log(f"μ1 (continuous)     = {mu1 * 1e3:.3f} mPa·s")
    log(f"μ2 (dispersed)      = {mu2 * 1e3:.3f} mPa·s")
    log(f"μ1/μ2               = {mu1 / mu2:.3f}")
    log("")

    # ------------------------------------------------------------------
    # 1. Analytical field vs Chebyshev
    # ------------------------------------------------------------------
    Q1, Q2 = mlh(10), mlh(1)
    Rd = 0.6 * R
    fld = solve_analytical(R, Rd, mu1, mu2, Q1, Q2)
    chb = solve_chebyshev(R, Rd, mu1, mu2, Q1, Q2, N=50)
    Q1r, Q2r = fld.reconstruct_Q()
    err1 = float(np.max(np.abs(chb.u1 - fld.u1(chb.r1))))
    err2 = float(np.max(np.abs(chb.u2[1:] - fld.u2(chb.r2[1:]))))

    log("1. Paper case  Q1=10 ml/h, Q2=1 ml/h, dd=0.6 dc")
    log(f"   ∂p1/∂x = {fld.P1:.4e} Pa/m")
    log(f"   ∂p2/∂x = {fld.P2:.4e} Pa/m")
    log(f"   U_tp   = {fld.Utp:.6f} m/s")
    log(f"   u2(0)  = {fld.u2(0.0):.6f} m/s")
    log(f"   Q1 recovered rel. error = {(Q1r - Q1) / Q1:.3e}")
    log(f"   Q2 recovered rel. error = {(Q2r - Q2) / Q2:.3e}")
    log(f"   Chebyshev max |u1| error = {err1:.3e}")
    log(f"   Chebyshev max |u2| error = {err2:.3e}")
    log(f"   Chebyshev P1 = {chb.P1:.4e}  (analytical {fld.P1:.4e})")
    log(f"   Chebyshev P2 = {chb.P2:.4e}  (analytical {fld.P2:.4e})")
    log("")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    r = np.linspace(1e-9, R, 400)
    ax.plot(r * 1e6, fld.u(r), "k-", lw=1.6, label="analytical")
    ax.plot(chb.r2 * 1e6, chb.u2, "o", ms=3.5, color="C0", label="Chebyshev $u_2$")
    ax.plot(chb.r1 * 1e6, chb.u1, "o", ms=3.5, color="C1", label="Chebyshev $u_1$")
    ax.axvline(Rd * 1e6, color="r", ls="--", lw=0.9, label="interface")
    ax.set_xlabel(r"Radial position $r$ (μm)")
    ax.set_ylabel(r"Axial velocity $u$ (m/s)")
    ax.set_title(r"Chebyshev vs analytical  ($Q_1=10$, $Q_2=1$ ml/h, $d_d=0.6 d_c$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "01_chebyshev_vs_analytical.png")

    # ------------------------------------------------------------------
    # 2. Fig. 3 — droplet growth
    # ------------------------------------------------------------------
    log("2. Fig. 3 style — velocity as the droplet grows (Q1=10, Q2=1 ml/h)")
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.5), sharey=True)
    growth_rows = []
    for ax, frac in zip(axes, [0.2, 0.4, 0.6, 0.8]):
        f = solve_analytical(R, frac * R, mu1, mu2, mlh(10), mlh(1))
        r_in = np.linspace(0.0, f.Rd, 220)
        r_out = np.linspace(f.Rd, f.R, 220)
        ax.plot(r_in * 1e6, f.u2(r_in), color="C0", label="droplet")
        ax.plot(r_out * 1e6, f.u1(r_out), color="C1", label="film")
        ax.axvline(f.Rd * 1e6, color="k", ls="--", lw=0.8)
        ax.set_title(rf"$d_d={frac:.1f}d_c$")
        ax.set_xlabel(r"$r$ (μm)")
        ax.grid(True, alpha=0.3)
        growth_rows.append((frac, f.u2(0.0), f.Utp, f.P1, f.P2))
        log(
            f"   dd={frac:.1f} dc   u(0)={f.u2(0.0): .5f} m/s   "
            f"Utp={f.Utp:.5f} m/s   P1={f.P1:.3e}   P2={f.P2:.3e}"
        )
    axes[0].set_ylabel(r"$u_x$ (m/s)")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Fig. 3: $Q_1=10$ ml/h, $Q_2=1$ ml/h", y=1.04)
    save(fig, "02_fig3_droplet_growth.png")
    log("")

    with (OUT / "fig3_growth.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dd/dc", "u_center_m_s", "Utp_m_s", "dp1_dx_Pa_m", "dp2_dx_Pa_m"])
        w.writerows(growth_rows)

    # ------------------------------------------------------------------
    # 3. Fig. 4a — Q1 sweep
    # ------------------------------------------------------------------
    log("3. Fig. 4a style — effect of Q1 (Q2=5 ml/h, dd=0.6 dc)")
    Rd = 0.6 * R
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    q1_rows = []
    for q1, c in zip([1, 5, 10], ["C0", "C1", "C2"]):
        f = solve_analytical(R, Rd, mu1, mu2, mlh(q1), mlh(5))
        r_in = np.linspace(0.0, Rd, 250)
        r_out = np.linspace(Rd, R, 250)
        ax.plot(
            np.r_[r_in, r_out] * 1e6,
            np.r_[f.u2(r_in), f.u1(r_out)],
            color=c,
            label=rf"$Q_1={q1}$ ml/h",
        )
        q1_rows.append((q1, f.u2(0.0), f.Utp, f.P1, f.P2))
        log(
            f"   Q1={q1:2d} ml/h   u(0)={f.u2(0.0): .5f}   Utp={f.Utp:.5f}   "
            f"P1={f.P1:.3e}   P2={f.P2:.3e}"
        )
    ax.axvline(Rd * 1e6, color="r", ls="--", lw=0.9, label="interface")
    ax.set_xlabel(r"Radial position $r$ (μm)")
    ax.set_ylabel(r"Axial velocity $u$ (m/s)")
    ax.set_title(r"Fig. 4a: $Q_2=5$ ml/h, $d_d=0.6 d_c$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "03_fig4a_Q1_sweep.png")
    log("")

    with (OUT / "fig4a_Q1_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Q1_mlh", "u_center_m_s", "Utp_m_s", "dp1_dx_Pa_m", "dp2_dx_Pa_m"])
        w.writerows(q1_rows)

    # ------------------------------------------------------------------
    # 4. Fig. 4b — Q2 sweep
    # ------------------------------------------------------------------
    log("4. Fig. 4b style — effect of Q2 (Q1=5 ml/h, dd=0.6 dc)")
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    q2_rows = []
    for q2, c in zip([1, 5, 10], ["C0", "C1", "C2"]):
        f = solve_analytical(R, Rd, mu1, mu2, mlh(5), mlh(q2))
        r_in = np.linspace(0.0, Rd, 250)
        r_out = np.linspace(Rd, R, 250)
        ax.plot(
            np.r_[r_in, r_out] * 1e6,
            np.r_[f.u2(r_in), f.u1(r_out)],
            color=c,
            label=rf"$Q_2={q2}$ ml/h",
        )
        q2_rows.append((q2, f.u2(0.0), f.Utp, f.P1, f.P2))
        log(
            f"   Q2={q2:2d} ml/h   u(0)={f.u2(0.0): .5f}   Utp={f.Utp:.5f}   "
            f"P1={f.P1:.3e}   P2={f.P2:.3e}"
        )
    ax.axvline(Rd * 1e6, color="r", ls="--", lw=0.9, label="interface")
    ax.set_xlabel(r"Radial position $r$ (μm)")
    ax.set_ylabel(r"Axial velocity $u$ (m/s)")
    ax.set_title(r"Fig. 4b: $Q_1=5$ ml/h, $d_d=0.6 d_c$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "04_fig4b_Q2_sweep.png")
    log("")

    with (OUT / "fig4b_Q2_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Q2_mlh", "u_center_m_s", "Utp_m_s", "dp1_dx_Pa_m", "dp2_dx_Pa_m"])
        w.writerows(q2_rows)

    # ------------------------------------------------------------------
    # 5. Force balance vs droplet diameter
    # ------------------------------------------------------------------
    ch_fig8 = Channel(sigma=2.636e-3)
    Q1, Q2 = mlh(4.456), mlh(0.518)
    log("5. Force balance  Q1=4.456 ml/h, Q2=0.518 ml/h, σ=2.636 mN/m")
    dds = np.linspace(0.2 * ch_fig8.dc, 0.97 * ch_fig8.dc, 80)
    Fr, Fs, Fdp, Ftot = [], [], [], []
    force_rows = []
    for dd in dds:
        f = solve_analytical(ch_fig8.R, 0.5 * dd, ch_fig8.mu1, ch_fig8.mu2, Q1, Q2)
        F = forces_on_droplet(f, ch_fig8.sigma)
        Fr.append(F.Fr)
        Fs.append(F.Fs)
        Fdp.append(F.Fdp)
        Ftot.append(F.Ftot)
        force_rows.append((dd * 1e6, F.Fr * 1e6, F.Fs * 1e6, F.Fdp * 1e6, F.Ftot * 1e6))

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(dds * 1e6, np.array(Fr) * 1e6, label=r"$F_r$ (holding)")
    ax.plot(dds * 1e6, np.array(Fs) * 1e6, label=r"$F_s$ (shear)")
    ax.plot(dds * 1e6, np.array(Fdp) * 1e6, label=r"$F_{\Delta p}$")
    ax.plot(dds * 1e6, np.array(Ftot) * 1e6, "k", label=r"$F_{\mathrm{tot}}$")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"Droplet diameter $d_d$ (μm)")
    ax.set_ylabel(r"Force (μN)")
    ax.set_title(r"Force balance vs $d_d$  ($Q_1=4.456$, $Q_2=0.518$ ml/h)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "05_force_balance.png")

    deq = equilibrium_diameter(ch_fig8, Q1, Q2)
    dcrit = critical_diameter(ch_fig8.dc, Q1, Q2)
    log(f"   d_eq   = {deq * 1e6:.2f} μm" if deq else "   d_eq not found")
    log(f"   d_crit = {dcrit * 1e6:.2f} μm")
    if deq is not None:
        dreal = real_diameter(deq, ch_fig8.dc, Q1, Q2)
        regime = "slug flow (d_eq > d_crit)" if deq > dcrit else "drip flow (d_eq < d_crit)"
        log(f"   d_real = {dreal * 1e6:.2f} μm")
        log(f"   regime = {regime}")
    log("")

    with (OUT / "force_balance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dd_um", "Fr_uN", "Fs_uN", "Fdp_uN", "Ftot_uN"])
        w.writerows(force_rows)

    # ------------------------------------------------------------------
    # 6. Diameters vs Q1 (Fig. 8)
    # ------------------------------------------------------------------
    log("6. Fig. 8 style — diameters vs Q1 (Q2=0.518 ml/h)")
    Q2 = mlh(0.518)
    Q1s = np.linspace(1.0, 12.0, 24)
    deqs, dcrits, dreals = [], [], []
    diam_rows = []
    for q1_mlh in Q1s:
        q1 = mlh(q1_mlh)
        deq_i = equilibrium_diameter(ch_fig8, q1, Q2)
        dc_i = critical_diameter(ch_fig8.dc, q1, Q2)
        deq_um = None if deq_i is None else deq_i * 1e6
        dreal_um = None if deq_i is None else real_diameter(deq_i, ch_fig8.dc, q1, Q2) * 1e6
        deqs.append(deq_um)
        dcrits.append(dc_i * 1e6)
        dreals.append(dreal_um)
        diam_rows.append(
            (
                q1_mlh,
                "" if deq_um is None else f"{deq_um:.4f}",
                f" {dc_i * 1e6:.4f}",
                "" if dreal_um is None else f"{dreal_um:.4f}",
            )
        )
        if abs(q1_mlh - 4.456) < 0.3 or q1_mlh in (1.0, 12.0):
            log(
                f"   Q1={q1_mlh:5.2f}  d_eq={deq_um}  d_crit={dc_i * 1e6:.1f}  d_real={dreal_um}"
            )

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(Q1s, deqs, "o-", ms=4, label=r"$d_{\mathrm{eq}}$")
    ax.plot(Q1s, dcrits, "s-", ms=4, label=r"$d_{\mathrm{crit}}$")
    ax.plot(Q1s, dreals, "^-", ms=4, label=r"$d_{\mathrm{real}}$")
    ax.axhline(ch_fig8.dc * 1e6, color="k", ls="--", lw=0.8, label=r"$d_c$")
    ax.set_xlabel(r"$Q_1$ (ml/h)")
    ax.set_ylabel(r"Diameter (μm)")
    ax.set_title(r"Fig. 8: $Q_2=0.518$ ml/h, $\sigma=2.636$ mN/m")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "06_fig8_diameters_vs_Q1.png")
    log("")

    with (OUT / "fig8_diameters_vs_Q1.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Q1_mlh", "deq_um", "dcrit_um", "dreal_um"])
        w.writerows(diam_rows)

    summary_path = OUT / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    log(f"Wrote {summary_path}")
    log("Done.")


if __name__ == "__main__":
    main()
