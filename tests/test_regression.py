"""Lock numerical results against the pre-refactor solver."""

from __future__ import annotations

import numpy as np
import pytest

from droplet_formation import (
    Channel,
    critical_diameter,
    equilibrium_diameter,
    mlh,
    real_diameter,
    solve_analytical,
    solve_chebyshev,
)

CHANNEL = Channel()
Q1_REF = mlh(10)
Q2_REF = mlh(1)
RD_REF = 0.6 * CHANNEL.R

# Values captured from the original implementation before the refactor.
P1_REF = -207608.7433147007
P2_REF = 66582.80533642447
UTP_REF = 0.0804160700915507
U2_0_REF = -0.05858411630665441
DEQ_REF = 0.0002891020627780227
DCRIT_REF = 0.00027591669750030455
DREAL_REF = 0.00031126643351205203

GROWTH = {
    0.2: (0.10738968227863367, 0.08909790178543463, -30469.252957334364, -78857.89812623589),
    0.4: (-0.010461253054652947, 0.059583149070670034, -83876.65262753541, 75492.30006840362),
    0.6: (-0.05858411630665441, 0.0804160700915507, -207608.7433147007, 66582.80533642447),
    0.8: (-0.14169734563025702, 0.15397781963425483, -1130367.0990362891, 79668.03064071572),
}


def test_reference_analytical_field() -> None:
    fld = solve_analytical(CHANNEL.R, RD_REF, CHANNEL.mu1, CHANNEL.mu2, Q1_REF, Q2_REF)
    assert fld.P1 == pytest.approx(P1_REF, rel=0, abs=1e-8)
    assert fld.P2 == pytest.approx(P2_REF, rel=0, abs=1e-8)
    assert fld.Utp == pytest.approx(UTP_REF, rel=0, abs=1e-14)
    assert float(np.asarray(fld.u2(0.0))) == pytest.approx(U2_0_REF, rel=0, abs=1e-14)


def test_flow_rates_round_trip() -> None:
    fld = solve_analytical(CHANNEL.R, RD_REF, CHANNEL.mu1, CHANNEL.mu2, Q1_REF, Q2_REF)
    Q1r, Q2r = fld.reconstruct_Q()
    assert Q1r == pytest.approx(Q1_REF, rel=1e-12)
    assert Q2r == pytest.approx(Q2_REF, rel=1e-12)


def test_boundary_conditions() -> None:
    fld = solve_analytical(CHANNEL.R, RD_REF, CHANNEL.mu1, CHANNEL.mu2, Q1_REF, Q2_REF)
    assert float(np.asarray(fld.u1(CHANNEL.R))) == pytest.approx(0.0, abs=1e-12)
    assert float(np.asarray(fld.u1(RD_REF))) == pytest.approx(fld.Utp, abs=1e-12)
    assert float(np.asarray(fld.u2(RD_REF))) == pytest.approx(fld.Utp, abs=1e-12)
    tau1 = CHANNEL.mu1 * float(np.asarray(fld.du1_dr(RD_REF)))
    tau2 = CHANNEL.mu2 * float(np.asarray(fld.du2_dr(RD_REF)))
    assert tau1 == pytest.approx(tau2, rel=1e-12)
    assert float(np.asarray(fld.du2_dr(0.0))) == pytest.approx(0.0, abs=1e-15)


def test_chebyshev_matches_analytical() -> None:
    fld = solve_analytical(CHANNEL.R, RD_REF, CHANNEL.mu1, CHANNEL.mu2, Q1_REF, Q2_REF)
    chb = solve_chebyshev(CHANNEL.R, RD_REF, CHANNEL.mu1, CHANNEL.mu2, Q1_REF, Q2_REF, N=50)
    err1 = float(np.max(np.abs(chb.u1 - fld.u1(chb.r1))))
    err2 = float(np.max(np.abs(chb.u2[1:] - fld.u2(chb.r2[1:]))))
    assert err1 < 1e-12
    assert err2 < 1e-12
    assert chb.P1 == pytest.approx(fld.P1, rel=1e-11)
    assert chb.P2 == pytest.approx(fld.P2, rel=1e-11)
    assert chb.Utp == pytest.approx(fld.Utp, rel=1e-11)


def test_growth_profiles() -> None:
    for frac, (u0, utp, p1, p2) in GROWTH.items():
        fld = solve_analytical(
            CHANNEL.R, frac * CHANNEL.R, CHANNEL.mu1, CHANNEL.mu2, Q1_REF, Q2_REF
        )
        assert float(np.asarray(fld.u2(0.0))) == pytest.approx(u0, rel=1e-12)
        assert fld.Utp == pytest.approx(utp, rel=1e-12)
        assert fld.P1 == pytest.approx(p1, rel=1e-12)
        assert fld.P2 == pytest.approx(p2, rel=1e-12)


def test_force_balance_diameters() -> None:
    ch = Channel(sigma=2.636e-3)
    Q1, Q2 = mlh(4.456), mlh(0.518)
    deq = equilibrium_diameter(ch, Q1, Q2)
    dcrit = critical_diameter(ch.dc, Q1, Q2)
    assert deq is not None
    dreal = real_diameter(deq, ch.dc, Q1, Q2)
    assert deq == pytest.approx(DEQ_REF, rel=1e-12)
    assert dcrit == pytest.approx(DCRIT_REF, rel=1e-12)
    assert dreal == pytest.approx(DREAL_REF, rel=1e-12)
