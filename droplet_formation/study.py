"""Reference study: figures, tables, and the KaTeX HTML report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from droplet_formation.analytical import solve_analytical
from droplet_formation.collocation import solve_chebyshev
from droplet_formation.forces import (
    critical_diameter,
    equilibrium_diameter,
    flow_regime,
    forces_on_droplet,
    real_diameter,
)
from droplet_formation.geometry import Channel
from droplet_formation.report import ReportCase, ReportSizes, write_html_report
from droplet_formation.units import mlh

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


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_reference_study(output_dir: Path | None = None) -> Path:
    """Reproduce the reference hydrodynamic cases and write artefacts."""
    output_dir = Path(output_dir) if output_dir else repository_root() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        lines.append(message)

    ch = Channel()
    R, mu1, mu2 = ch.R, ch.mu1, ch.mu2
    log("Droplet-formation solver results")
    log("=" * 60)
    log(f"Channel diameter dc = {ch.dc * 1e6:.1f} μm")
    log(f"Channel radius   R  = {R * 1e6:.1f} μm")
    log(f"μ1 (continuous)     = {mu1 * 1e3:.3f} mPa·s")
    log(f"μ2 (dispersed)      = {mu2 * 1e3:.3f} mPa·s")
    log(f"μ1/μ2               = {mu1 / mu2:.3f}")
    log("")

    Q1, Q2 = mlh(10), mlh(1)
    Rd = 0.6 * R
    fld = solve_analytical(R, Rd, mu1, mu2, Q1, Q2)
    chb = solve_chebyshev(R, Rd, mu1, mu2, Q1, Q2, N=50)
    Q1r, Q2r = fld.reconstruct_Q()
    err1 = float(np.max(np.abs(chb.u1 - fld.u1(chb.r1))))
    err2 = float(np.max(np.abs(chb.u2[1:] - fld.u2(chb.r2[1:]))))
    u2_0 = float(np.asarray(fld.u2(0.0)))
    q1_rel = float((Q1r - Q1) / Q1)
    q2_rel = float((Q2r - Q2) / Q2)

    log("1. Reference case  Q1=10 ml/h, Q2=1 ml/h, dd=0.6 dc")
    log(f"   ∂p1/∂x = {fld.P1:.4e} Pa/m")
    log(f"   ∂p2/∂x = {fld.P2:.4e} Pa/m")
    log(f"   U_tp   = {fld.Utp:.6f} m/s")
    log(f"   u2(0)  = {u2_0:.6f} m/s")
    log(f"   Q1 recovered rel. error = {q1_rel:.3e}")
    log(f"   Q2 recovered rel. error = {q2_rel:.3e}")
    log(f"   Chebyshev max |u1| error = {err1:.3e}")
    log(f"   Chebyshev max |u2| error = {err2:.3e}")
    log(f"   Chebyshev P1 = {chb.P1:.4e}  (analytical {fld.P1:.4e})")
    log(f"   Chebyshev P2 = {chb.P2:.4e}  (analytical {fld.P2:.4e})")
    log("")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    r = np.linspace(1e-9, R, 400)
    ax.plot(r * 1e6, fld.u(r), "k-", lw=1.6, label="analytical")
    ax.plot(chb.r2 * 1e6, chb.u2, "o", ms=3.5, color="C0", label=r"Chebyshev $u_2$")
    ax.plot(chb.r1 * 1e6, chb.u1, "o", ms=3.5, color="C1", label=r"Chebyshev $u_1$")
    ax.axvline(Rd * 1e6, color="r", ls="--", lw=0.9, label="interface")
    ax.set_xlabel(r"Radial position $r$ (μm)")
    ax.set_ylabel(r"Axial velocity $u$ (m/s)")
    ax.set_title(r"Chebyshev vs analytical  ($Q_1=10$, $Q_2=1$ ml/h, $d_d=0.6 d_c$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, "01_chebyshev_vs_analytical.png")

    log("2. Velocity as the droplet grows (Q1=10, Q2=1 ml/h)")
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.5), sharey=True)
    growth_rows: list[tuple] = []
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
        growth_rows.append((frac, float(np.asarray(f.u2(0.0))), f.Utp, f.P1, f.P2))
        log(
            f"   dd={frac:.1f} dc   u(0)={float(np.asarray(f.u2(0.0))): .5f} m/s   "
            f"Utp={f.Utp:.5f} m/s   P1={f.P1:.3e}   P2={f.P2:.3e}"
        )
    axes[0].set_ylabel(r"$u_x$ (m/s)")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"$Q_1=10$ ml/h, $Q_2=1$ ml/h", y=1.04)
    _save(fig, output_dir, "02_droplet_growth.png")
    log("")
    _write_csv(
        output_dir / "droplet_growth.csv",
        ["dd/dc", "u_center_m_s", "Utp_m_s", "dp1_dx_Pa_m", "dp2_dx_Pa_m"],
        growth_rows,
    )

    log("3. Effect of Q1 (Q2=5 ml/h, dd=0.6 dc)")
    Rd = 0.6 * R
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    q1_rows: list[tuple] = []
    for q1, color in zip([1, 5, 10], ["C0", "C1", "C2"]):
        f = solve_analytical(R, Rd, mu1, mu2, mlh(q1), mlh(5))
        r_in = np.linspace(0.0, Rd, 250)
        r_out = np.linspace(Rd, R, 250)
        ax.plot(
            np.r_[r_in, r_out] * 1e6,
            np.r_[f.u2(r_in), f.u1(r_out)],
            color=color,
            label=rf"$Q_1={q1}$ ml/h",
        )
        q1_rows.append((q1, float(np.asarray(f.u2(0.0))), f.Utp, f.P1, f.P2))
        log(
            f"   Q1={q1:2d} ml/h   u(0)={float(np.asarray(f.u2(0.0))): .5f}   "
            f"Utp={f.Utp:.5f}   P1={f.P1:.3e}   P2={f.P2:.3e}"
        )
    ax.axvline(Rd * 1e6, color="r", ls="--", lw=0.9, label="interface")
    ax.set_xlabel(r"Radial position $r$ (μm)")
    ax.set_ylabel(r"Axial velocity $u$ (m/s)")
    ax.set_title(r"$Q_2=5$ ml/h, $d_d=0.6 d_c$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, "03_Q1_sweep.png")
    log("")
    _write_csv(
        output_dir / "Q1_sweep.csv",
        ["Q1_mlh", "u_center_m_s", "Utp_m_s", "dp1_dx_Pa_m", "dp2_dx_Pa_m"],
        q1_rows,
    )

    log("4. Effect of Q2 (Q1=5 ml/h, dd=0.6 dc)")
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    q2_rows: list[tuple] = []
    for q2, color in zip([1, 5, 10], ["C0", "C1", "C2"]):
        f = solve_analytical(R, Rd, mu1, mu2, mlh(5), mlh(q2))
        r_in = np.linspace(0.0, Rd, 250)
        r_out = np.linspace(Rd, R, 250)
        ax.plot(
            np.r_[r_in, r_out] * 1e6,
            np.r_[f.u2(r_in), f.u1(r_out)],
            color=color,
            label=rf"$Q_2={q2}$ ml/h",
        )
        q2_rows.append((q2, float(np.asarray(f.u2(0.0))), f.Utp, f.P1, f.P2))
        log(
            f"   Q2={q2:2d} ml/h   u(0)={float(np.asarray(f.u2(0.0))): .5f}   "
            f"Utp={f.Utp:.5f}   P1={f.P1:.3e}   P2={f.P2:.3e}"
        )
    ax.axvline(Rd * 1e6, color="r", ls="--", lw=0.9, label="interface")
    ax.set_xlabel(r"Radial position $r$ (μm)")
    ax.set_ylabel(r"Axial velocity $u$ (m/s)")
    ax.set_title(r"$Q_1=5$ ml/h, $d_d=0.6 d_c$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, "04_Q2_sweep.png")
    log("")
    _write_csv(
        output_dir / "Q2_sweep.csv",
        ["Q2_mlh", "u_center_m_s", "Utp_m_s", "dp1_dx_Pa_m", "dp2_dx_Pa_m"],
        q2_rows,
    )

    ch_sigma = Channel(sigma=2.636e-3)
    Q1, Q2 = mlh(4.456), mlh(0.518)
    log("5. Force balance  Q1=4.456 ml/h, Q2=0.518 ml/h, σ=2.636 mN/m")
    dds = np.linspace(0.2 * ch_sigma.dc, 0.97 * ch_sigma.dc, 80)
    Fr, Fs, Fdp, Ftot = [], [], [], []
    force_rows: list[tuple] = []
    for dd in dds:
        f = solve_analytical(ch_sigma.R, 0.5 * dd, ch_sigma.mu1, ch_sigma.mu2, Q1, Q2)
        force = forces_on_droplet(f, ch_sigma.sigma)
        Fr.append(force.Fr)
        Fs.append(force.Fs)
        Fdp.append(force.Fdp)
        Ftot.append(force.Ftot)
        force_rows.append(
            (dd * 1e6, force.Fr * 1e6, force.Fs * 1e6, force.Fdp * 1e6, force.Ftot * 1e6)
        )

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
    _save(fig, output_dir, "05_force_balance.png")

    deq = equilibrium_diameter(ch_sigma, Q1, Q2)
    dcrit = critical_diameter(ch_sigma.dc, Q1, Q2)
    log(f"   d_eq   = {deq * 1e6:.2f} μm" if deq else "   d_eq not found")
    log(f"   d_crit = {dcrit * 1e6:.2f} μm")
    dreal = None
    regime = None
    if deq is not None:
        dreal = real_diameter(deq, ch_sigma.dc, Q1, Q2)
        regime = flow_regime(deq, dcrit)
        log(f"   d_real = {dreal * 1e6:.2f} μm")
        log(f"   regime = {regime} flow")
    log("")
    _write_csv(
        output_dir / "force_balance.csv",
        ["dd_um", "Fr_uN", "Fs_uN", "Fdp_uN", "Ftot_uN"],
        force_rows,
    )

    log("6. Diameters vs Q1 (Q2=0.518 ml/h)")
    Q2 = mlh(0.518)
    Q1s = np.linspace(1.0, 12.0, 24)
    deqs, dcrits, dreals = [], [], []
    diam_rows: list[tuple] = []
    for q1_mlh in Q1s:
        q1 = mlh(q1_mlh)
        deq_i = equilibrium_diameter(ch_sigma, q1, Q2)
        dc_i = critical_diameter(ch_sigma.dc, q1, Q2)
        deq_um = None if deq_i is None else deq_i * 1e6
        dreal_um = None if deq_i is None else real_diameter(deq_i, ch_sigma.dc, q1, Q2) * 1e6
        deqs.append(deq_um)
        dcrits.append(dc_i * 1e6)
        dreals.append(dreal_um)
        diam_rows.append(
            (
                q1_mlh,
                "" if deq_um is None else f"{deq_um:.4f}",
                f"{dc_i * 1e6:.4f}",
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
    ax.axhline(ch_sigma.dc * 1e6, color="k", ls="--", lw=0.8, label=r"$d_c$")
    ax.set_xlabel(r"$Q_1$ (ml/h)")
    ax.set_ylabel(r"Diameter (μm)")
    ax.set_title(r"$Q_2=0.518$ ml/h, $\sigma=2.636$ mN/m")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, "06_diameters_vs_Q1.png")
    log("")
    _write_csv(
        output_dir / "diameters_vs_Q1.csv",
        ["Q1_mlh", "deq_um", "dcrit_um", "dreal_um"],
        diam_rows,
    )

    html_path = write_html_report(
        output_dir,
        ReportCase(
            P1=float(fld.P1),
            P2=float(fld.P2),
            Utp=float(fld.Utp),
            u2_0=u2_0,
            err_u1=err1,
            err_u2=err2,
            q1_rel=q1_rel,
            q2_rel=q2_rel,
        ),
        ReportSizes(
            deq_um=None if deq is None else deq * 1e6,
            dcrit_um=dcrit * 1e6,
            dreal_um=None if dreal is None else dreal * 1e6,
            regime=regime,
        ),
    )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {summary_path}")
    log(f"Wrote {html_path}")
    log("Done.")
    return output_dir
