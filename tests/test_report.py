"""KaTeX report contains math delimiters and the renderer assets."""

from __future__ import annotations

from pathlib import Path

from droplet_formation.report import ReportCase, ReportSizes, write_html_report


def test_html_report_includes_katex_pipeline(tmp_path: Path) -> None:
    path = write_html_report(
        tmp_path,
        ReportCase(
            P1=-1.0,
            P2=1.0,
            Utp=0.08,
            u2_0=-0.05,
            err_u1=1e-14,
            err_u2=1e-14,
            q1_rel=1e-15,
            q2_rel=1e-15,
        ),
        ReportSizes(deq_um=289.1, dcrit_um=275.9, dreal_um=311.3, regime="slug"),
    )
    html = path.read_text(encoding="utf-8")
    assert "katex.min.js" in html
    assert "auto-render.min.js" in html
    assert "renderMathInElement" in html
    assert r"\frac{1}{r}" in html
    assert "$$" in html
    assert "U_{tp}" in html
    assert "figures/02_droplet_growth.png" in html
    assert "figures/03_Q1_sweep.png" in html
    assert "figures/06_diameters_vs_Q1.png" in html
