"""
bench/figure.py — render the one benchmark figure to docs/benchmark.svg.

    python -m bench.figure

The figure shows the headline number in business language: how often each tool
blamed a cause that was not actually there, across the 45 no-single-cause
cases. Reads bench/metrics.json; the naive bar renders as "run with a DashScope
key" until that column has been filled.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "docs" / "benchmark.svg"

INK, MUTED, LINE = "#0f172a", "#64748b", "#e2e8f0"
GOOD, BAD, PEND = "#16a34a", "#dc2626", "#94a3b8"


def _bar(x, y, w, h, fill, label, value, sub):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}"/>'
        f'<text x="{x + w + 12}" y="{y + h/2 + 5}" font-size="15" fill="{INK}" '
        f'font-weight="600">{value}</text>'
        f'<text x="{x}" y="{y - 8}" font-size="13.5" fill="{INK}">{label}</text>'
        f'<text x="{x}" y="{y + h + 18}" font-size="12" fill="{MUTED}">{sub}</text>')


def render() -> str:
    m = json.loads((HERE / "metrics.json").read_text())
    ours = m["prove_or_abstain"]
    naive = m.get("naive_qwen")

    W, H = 640, 300
    x0, top, barmax, bh = 40, 90, 380, 46
    n_no = ours["n_no_cause"]

    svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">']
    svg.append(f'<text x="{x0}" y="34" font-size="18" font-weight="700" '
               f'fill="{INK}">How often each tool blamed a cause that wasn\'t there</text>')
    svg.append(f'<text x="{x0}" y="56" font-size="13" fill="{MUTED}">'
               f'across {n_no} cases where no single segment was actually responsible '
               f'— lower is better</text>')

    r = ours["cause_invention_rate"]
    svg.append(_bar(x0, top, max(barmax * r, 3), bh, GOOD if r < 0.1 else BAD,
                    "prove-or-abstain", f"{r:.0%}",
                    f"{ours['invented_count']} of {n_no} — it refuses when the data is unclear"))

    y2 = top + bh + 46
    if naive:
        rn = naive["cause_invention_rate"]
        svg.append(_bar(x0, y2, max(barmax * rn, 3), bh, BAD if rn >= 0.1 else GOOD,
                        "plain Qwen prompt", f"{rn:.0%}",
                        f"{naive['invented_count']} of {n_no} — same model, no discipline"))
    else:
        svg.append(f'<rect x="{x0}" y="{y2}" width="{barmax}" height="{bh}" rx="4" '
                   f'fill="none" stroke="{PEND}" stroke-width="1.5" stroke-dasharray="6 5"/>')
        svg.append(f'<text x="{x0}" y="{y2 - 8}" font-size="13.5" fill="{INK}">'
                   f'plain Qwen prompt</text>')
        svg.append(f'<text x="{x0 + 16}" y="{y2 + bh/2 + 5}" font-size="13" fill="{MUTED}">'
                   f'run <tspan font-family="monospace">python -m bench.run</tspan> with a '
                   f'DashScope key to fill this bar</text>')

    svg.append(f'<text x="{x0}" y="{H - 14}" font-size="11.5" fill="{MUTED}">'
               f'Both tools run on the same Qwen model; the difference is what each is '
               f'allowed to decide.</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
