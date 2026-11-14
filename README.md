# Droplet Formation in Microfluidic Channels

Axisymmetric Stokes-flow model for a growing droplet in a cylindrical microchannel. The solver returns equatorial velocity profiles, interfacial shear, pressure gradients, and the force balance that sets droplet size and the drip–slug transition.

The formulation follows Svetlov & Abiev, *Chem. Eng. Sci.* **235**, 116493 (2021).

## Model

A dispersed-phase droplet (phase 2) occupies \(0 \le r \le R_d\). A continuous-phase film (phase 1) occupies \(R_d \le r \le R\). Under creeping, steady, axisymmetric flow the axial momentum equation reduces to

\[
\frac{1}{r}\frac{\mathrm{d}}{\mathrm{d}r}\left(r\frac{\mathrm{d}u_i}{\mathrm{d}r}\right)
= \frac{1}{\mu_i}\frac{\partial p_i}{\partial x}.
\]

Boundary conditions:

- no-slip at the wall, \(u_1(R)=0\)
- symmetry on the axis, \(u_2'(0)=0\)
- velocity continuity at the interface, \(u_1(R_d)=u_2(R_d)=U_{tp}\)
- shear continuity, \(\mu_1 u_1'(R_d)=\mu_2 u_2'(R_d)\)

The two pressure gradients are **not** equal. They are fixed by the imposed volumetric flow rates

\[
Q_1 = 2\pi\int_{R_d}^{R} u_1 r\,\mathrm{d}r,\qquad
Q_2 = 2\pi\int_{0}^{R_d} u_2 r\,\mathrm{d}r.
\]

Detachment is predicted from the equatorial force balance

\[
F_{\mathrm{tot}} = F_{\Delta p} + F_s - F_r = 0,
\]

with interfacial tension \(F_r=\pi d_d\sigma\), shear \(F_s=\tau(R_d)A_d\), and pressure force \(F_{\Delta p}=(\partial p_2/\partial x)V_d\). Equilibrium, critical, and real diameters follow from that balance and Cubaud’s neck-collapse time.

Default fluid properties match the paper’s water–cyclohexane micromixer: \(d_c=300\,\mu\mathrm{m}\), \(\mu_1=2.24\,\mathrm{mPa\cdot s}\), \(\mu_2=0.97\,\mathrm{mPa\cdot s}\).

## Features

- Closed-form two-phase velocity field (integration constants from BCs + \(Q_1,Q_2\))
- Chebyshev–Gauss–Lobatto collocation of the same Stokes problem, with \(\partial p_1/\partial x\) and \(\partial p_2/\partial x\) as additional unknowns
- Clenshaw–Curtis quadrature for the flow-rate constraints
- Force-balance scan for \(d_{\mathrm{eq}}\), \(d_{\mathrm{crit}}\), \(d_{\mathrm{real}}\)
- Spectral kernel in `Project/ChebMultiPhase.ipynb` for the differentiation matrix and two-domain collocation maps

## Repository

```
svetlov_model.py              hydrodynamic model (analytical + spectral + forces)
Svetlov_Abiev_solver.ipynb    interactive walkthrough
run_and_save_results.py       batch run; writes figures and tables
results/                      generated profiles, force curves, diameter maps
Project/ChebMultiPhase.ipynb  Chebyshev collocation kernel
```

Core API (`Svetlov_Abiev_model/svetlov_model.py`):

```python
from svetlov_model import Channel, mlh, solve_analytical, solve_chebyshev
from svetlov_model import forces_on_droplet, equilibrium_diameter, critical_diameter

ch = Channel()                          # dc, μ1, μ2, σ
fld = solve_analytical(ch.R, Rd, ch.mu1, ch.mu2, Q1, Q2)
chb = solve_chebyshev(ch.R, Rd, ch.mu1, ch.mu2, Q1, Q2, N=50)
```

`solve_analytical` returns \(u_1(r)\), \(u_2(r)\), \(U_{tp}\), and both pressure gradients. `solve_chebyshev` discretizes the identical BVP on \(N+1\) Gauss–Lobatto nodes per phase.

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

## Quick start

```bash
cd Svetlov_Abiev_model
python3 run_and_save_results.py
```

Outputs land in `Svetlov_Abiev_model/results/`:

| File | Contents |
|---|---|
| `01_chebyshev_vs_analytical.png` | Spectral vs closed-form velocity |
| `02_fig3_droplet_growth.png` | Profiles during droplet growth |
| `03_fig4a_Q1_sweep.png` | Effect of continuous-phase flow rate |
| `04_fig4b_Q2_sweep.png` | Effect of dispersed-phase flow rate |
| `05_force_balance.png` | \(F_r\), \(F_s\), \(F_{\Delta p}\) vs droplet diameter |
| `06_fig8_diameters_vs_Q1.png` | \(d_{\mathrm{eq}}\), \(d_{\mathrm{crit}}\), \(d_{\mathrm{real}}\) vs \(Q_1\) |
| `*.csv`, `summary.txt` | Numeric tables |

Alternatively open `Svetlov_Abiev_solver.ipynb` and run all cells.

Minimal example:

```python
import sys
sys.path.insert(0, "Svetlov_Abiev_model")

from svetlov_model import Channel, mlh, solve_analytical

ch = Channel()
Rd = 0.6 * ch.R
fld = solve_analytical(ch.R, Rd, ch.mu1, ch.mu2, mlh(10), mlh(1))

print(fld.P1, fld.P2, fld.Utp)
print(fld.u2(0.0))          # centreline velocity
print(fld.reconstruct_Q())  # should recover Q1, Q2
```

## Results

For \(Q_1=10\,\mathrm{ml/h}\), \(Q_2=1\,\mathrm{ml/h}\), \(d_d=0.6\,d_c\):

| Quantity | Value |
|---|---|
| \(\partial p_1/\partial x\) | \(-2.08\times 10^{5}\) Pa/m |
| \(\partial p_2/\partial x\) | \(+6.66\times 10^{4}\) Pa/m |
| Interface speed \(U_{tp}\) | \(0.0804\) m/s |
| Centreline velocity \(u_2(0)\) | \(-0.0586\) m/s |

The two pressure gradients have opposite sign, so the droplet core reverses relative to the film (internal vortex). Raising \(Q_1\) accelerates the film and can invert the core; raising \(Q_2\) mainly changes the droplet interior and leaves the film almost unchanged.

Force balance at \(Q_1=4.456\,\mathrm{ml/h}\), \(Q_2=0.518\,\mathrm{ml/h}\), \(\sigma=2.636\,\mathrm{mN/m}\):

| Diameter | Value |
|---|---|
| \(d_{\mathrm{eq}}\) | \(289\,\mu\mathrm{m}\) |
| \(d_{\mathrm{crit}}\) | \(276\,\mu\mathrm{m}\) |
| \(d_{\mathrm{real}}\) | \(311\,\mu\mathrm{m}\) |

Here \(d_{\mathrm{eq}}>d_{\mathrm{crit}}\), so the detached object is a slug. Increasing \(Q_1\) lowers \(d_{\mathrm{eq}}\) and raises \(d_{\mathrm{crit}}\); the curves cross at the slug → drip boundary.

## Verification

On the case above, recovered flow rates and Chebyshev velocities agree with the closed form to \(\sim 10^{-14}\). For equal viscosities the field reduces to Hagen–Poiseuille \(u=\frac{1}{4\mu}(\partial p/\partial x)(r^2-R^2)\) when the two-phase constraints collapse to a single fluid.

The spectral differentiation matrix uses Trefethen’s Chebyshev–Gauss–Lobatto form, including the endpoint entries \(\pm(2N^2+1)/6\).

## Reference

S. D. Svetlov, R. Sh. Abiev, *Mathematical modeling of the droplet formation process in a microfluidic device*, Chemical Engineering Science **235** (2021) 116493. https://doi.org/10.1016/j.ces.2021.116493
