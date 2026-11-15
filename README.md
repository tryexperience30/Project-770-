# Droplet Formation in Microfluidic Channels

Axisymmetric Stokes-flow solver for a growing droplet in a cylindrical microchannel. The code computes equatorial velocity profiles, interfacial shear, independent phase pressure gradients, and the force balance that sets droplet size and the drip–slug transition.

The hydrodynamics follow Svetlov & Abiev, *Chemical Engineering Science* **235**, 116493 (2021).

## Quick Start

From the repository root:

```bash
python -m pip install -r requirements.txt
python run.py
```

The solver writes all outputs automatically. Open the HTML report in a browser:

```text
results/index.html
```

The page renders equations with KaTeX and embeds the generated figures. A text summary is saved at `results/summary.txt`.

To run the regression tests:

```bash
python -m pytest -q
```

## Overview

The solver evaluates two-phase creeping flow in a cylindrical channel. A dispersed-phase droplet occupies the core; a continuous-phase film occupies the annular gap. Closed-form Stokes profiles are verified by Chebyshev collocation. An equatorial force balance then estimates the equilibrium, critical, and real droplet diameters.

There is a single execution entry point: `run.py`. Scientific calculations live in the `droplet_formation` package and are not duplicated in the runner.

## Problem being investigated

A dispersed-phase droplet occupies $0 \le r \le R_d$ inside a continuous-phase film $R_d \le r \le R$. The questions addressed by the solver are:

- How do the imposed flow rates $Q_1$ and $Q_2$ shape the equatorial velocity field?
- When do detaching forces overcome interfacial tension?
- Is the detached object a spherical drop or a liquid slug?

## Physical and mathematical model

The flow is steady, axisymmetric, and inertialess. Gravity is neglected. Each phase has a constant viscosity, and the two axial pressure gradients $\partial p_1/\partial x$ and $\partial p_2/\partial x$ are independent. They are determined by the imposed volumetric flow rates rather than by a single shared pressure gradient.

The droplet is treated as spherical for the force analysis. Hydrodynamics are evaluated in the equatorial plane. Default fluid properties match a water–cyclohexane micromixer: $d_c=300\,\mu\mathrm{m}$, $\mu_1=2.24\,\mathrm{mPa\cdot s}$, $\mu_2=0.97\,\mathrm{mPa\cdot s}$.

## Governing equations

Under creeping, steady, axisymmetric flow the axial momentum equation reduces to

```math
\frac{1}{r}\frac{\mathrm{d}}{\mathrm{d}r}\left(r\frac{\mathrm{d}u_i}{\mathrm{d}r}\right)
=
\frac{1}{\mu_i}\frac{\partial p_i}{\partial x}
```

Boundary conditions:

- wall no-slip: $u_1(R)=0$
- centreline symmetry: $u_2'(0)=0$
- velocity continuity: $u_1(R_d)=u_2(R_d)=U_{tp}$
- shear continuity: $\mu_1 u_1'(R_d)=\mu_2 u_2'(R_d)$

The two pressure gradients are closed by the flow-rate integrals

```math
Q_1 = 2\pi\int_{R_d}^{R} u_1 r\,\mathrm{d}r,\qquad
Q_2 = 2\pi\int_{0}^{R_d} u_2 r\,\mathrm{d}r
```

Detachment is estimated from the equatorial force balance

```math
F_{\mathrm{tot}} = F_{\Delta p} + F_s - F_r = 0
```

with $F_r=\pi d_d\sigma$, $F_s=\tau(R_d)A_d$, and $F_{\Delta p}=(\partial p_2/\partial x)V_d$. Equilibrium, critical, and real diameters follow from that balance and the neck-collapse time $t_{\mathrm{col}}=d_c^3/Q_1$.

## Computational methodology

1. **Analytical field.** Integrate the Stokes operator, apply the four boundary conditions, and close the system with the two flow-rate integrals. The result is a $6\times 6$ linear solve for $P_1$, $P_2$, $U_{tp}$, $c_{11}$, $c_{21}$, and $c_{22}$.
2. **Chebyshev collocation.** Discretise the same boundary-value problem on Gauss–Lobatto nodes and recover $Q_1$ and $Q_2$ with Clenshaw–Curtis quadrature. This is an independent check of the closed form.
3. **Force scan.** Evaluate $F_{\mathrm{tot}}(d_d)$ and interpolate the zero to obtain $d_{\mathrm{eq}}$. Compare with $d_{\mathrm{crit}}$ to classify slug versus drip flow.

The spectral differentiation matrix is Trefethen’s Chebyshev–Gauss–Lobatto form, including the endpoint entries $\pm(2N^2+1)/6$.

`run.py` executes all three stages, writes figures and CSV tables, and generates the HTML report.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── run.py                 # only command a new user needs
├── droplet_formation/     # modular Stokes solver
│   ├── analytical.py      # closed-form velocity field
│   ├── collocation.py     # Chebyshev–Gauss–Lobatto discretisation
│   ├── forces.py          # equatorial force balance and diameters
│   ├── geometry.py        # channel and fluid properties
│   ├── study.py           # reference cases and artefact writer
│   └── report.py          # KaTeX HTML report
├── results/
│   ├── index.html         # open this file in a browser
│   ├── summary.txt
│   ├── figures/           # PNG plots
│   └── data/              # CSV tables
└── tests/                 # regression and validation
```

The package remains modular. `run.py` only orchestrates the existing solver; it does not re-implement the mathematics.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Runtime libraries are NumPy and Matplotlib. Pytest is included so the validation suite can be run with the same install step.

## Running the project

```bash
python run.py
```

This command:

- solves the analytical Stokes field
- solves the Chebyshev collocation problem
- evaluates the equatorial force balance
- computes $d_{\mathrm{eq}}$, $d_{\mathrm{crit}}$, and $d_{\mathrm{real}}$
- writes figures, CSV tables, `results/index.html`, and `results/summary.txt`
- prints the output locations on success

If a calculation error occurs, the process exits with a non-zero status and prints the exception.

Optional programmatic use of the same library:

```python
from droplet_formation import Channel, mlh, solve_analytical, solve_chebyshev

ch = Channel()
Rd = 0.6 * ch.R
fld = solve_analytical(ch.R, Rd, ch.mu1, ch.mu2, mlh(10), mlh(1))
chb = solve_chebyshev(ch.R, Rd, ch.mu1, ch.mu2, mlh(10), mlh(1), N=50)

print(fld.P1, fld.P2, fld.Utp)
print(float(fld.u2(0.0)))
```

## Generated outputs

After `python run.py`:

| Path | Contents |
|---|---|
| `results/index.html` | KaTeX results report |
| `results/summary.txt` | Readable numerical summary |
| `results/figures/01_chebyshev_vs_analytical.png` | Spectral vs closed-form velocity |
| `results/figures/02_droplet_growth.png` | Profiles during droplet growth |
| `results/figures/03_Q1_sweep.png` | Effect of $Q_1$ |
| `results/figures/04_Q2_sweep.png` | Effect of $Q_2$ |
| `results/figures/05_force_balance.png` | $F_r$, $F_s$, $F_{\Delta p}$ vs $d_d$ |
| `results/figures/06_diameters_vs_Q1.png` | $d_{\mathrm{eq}}$, $d_{\mathrm{crit}}$, $d_{\mathrm{real}}$ vs $Q_1$ |
| `results/data/*.csv` | Numeric tables for the figures above |

Open `results/index.html` in a browser. The page loads KaTeX from a CDN, so viewing the rendered equations requires network access once.

## Numerical validation

Reference hydrodynamic case, $Q_1=10\,\mathrm{ml/h}$, $Q_2=1\,\mathrm{ml/h}$, $d_d=0.6\,d_c$:

| Quantity | Value |
|---|---|
| $\partial p_1/\partial x$ | $-2.076\times 10^{5}\,\mathrm{Pa/m}$ |
| $\partial p_2/\partial x$ | $+6.658\times 10^{4}\,\mathrm{Pa/m}$ |
| $U_{tp}$ | $0.0804\,\mathrm{m/s}$ |
| $u_2(0)$ | $-0.0586\,\mathrm{m/s}$ |

The pressure gradients have opposite sign, so the droplet core reverses relative to the film. Recovered flow rates and Chebyshev velocities agree with the closed form to $\sim 10^{-14}$.

Force balance at $Q_1=4.456\,\mathrm{ml/h}$, $Q_2=0.518\,\mathrm{ml/h}$, $\sigma=2.636\,\mathrm{mN/m}$:

| Diameter | Value |
|---|---|
| $d_{\mathrm{eq}}$ | $289\,\mu\mathrm{m}$ |
| $d_{\mathrm{crit}}$ | $276\,\mu\mathrm{m}$ |
| $d_{\mathrm{real}}$ | $311\,\mu\mathrm{m}$ |

Here $d_{\mathrm{eq}}>d_{\mathrm{crit}}$, so the detached object is a slug. Increasing $Q_1$ lowers $d_{\mathrm{eq}}$ and raises $d_{\mathrm{crit}}$; the curves cross at the slug → drip boundary.

These values are locked by the regression tests in `tests/`.

## Model assumptions and limitations

- Steady creeping flow; inertia and gravity are neglected.
- The droplet is treated as spherical; hydrodynamics are evaluated only in the equatorial plane.
- Shear on the full surface is approximated by the equatorial value, so $F_s$ is an upper bound.
- $F_{\Delta p}$ uses $\partial p_2/\partial x$, not the local film gradient.
- No empirical correction factor is applied to $F_s$; sizes are the model as written, not a fitted flow map.
- Neck-collapse time follows Cubaud’s $t_{\mathrm{col}}=d_c^3/Q_1$.

## Scientific references

S. D. Svetlov, R. Sh. Abiev, *Mathematical modeling of the droplet formation process in a microfluidic device*, Chemical Engineering Science **235** (2021) 116493. https://doi.org/10.1016/j.ces.2021.116493

L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.

T. Cubaud and T. G. Mason, *Capillary threads and viscous droplets in square microchannels*, Physics of Fluids **20** (2008) 053302.
