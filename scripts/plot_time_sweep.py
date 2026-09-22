"""Figures for the precursor time sweep. Usage: plot_time_sweep.py <result dir>."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIXED = ("d2", "d1.5", "d1", "d0.75", "d0.5", "d0.25", "T0")
LEAD = {"d2": 2.0, "d1.5": 1.5, "d1": 1.0, "d0.75": 0.75, "d0.5": 0.5, "d0.25": 0.25, "T0": 0.02}
COLORS = {"real": "#b5471b", "match": "#3a6ea5", "sim": "#5a8a3a"}
LABELS = {
    "real": "real failed trajectory",
    "match": "successful trajectory, same position",
    "sim": "simulator-found failure",
}


def _ci(values):
    v = np.asarray(values)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def main(path: str) -> None:
    d = Path(path)
    r = json.loads((d / "stage1.json").read_text())
    sm = r["seed_means"]
    worlds = list(r["worlds"])
    out = d / "figures"
    out.mkdir(exist_ok=True)
    x = np.array([LEAD[k] for k in FIXED])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes[0, 0]
    for src in ("real", "match", "sim"):
        m, e = zip(*(_ci(list(sm[f"{src}_{k}"].values())) for k in FIXED))
        ax.errorbar(x, m, yerr=e, marker="o", color=COLORS[src], label=LABELS[src], capsize=3)
    for arm, ls in (("A", ":"), ("B", "--"), ("ADR", "-."), ("C", (0, (1, 3)))):
        ax.axhline(np.mean(list(sm[arm].values())), color="k", ls=ls, lw=1, label=arm)
    ent = np.mean(list(sm["real_entry"].values()))
    lead_entry = np.mean([r["setup"][w]["real"]["entry_lead_s"] for w in worlds])
    ax.plot(
        [lead_entry], [ent], "s", color=COLORS["real"], mfc="white", label="real entry (old FBR)"
    )
    ax.set_xlabel("replay start, seconds before failure onset")
    ax.set_ylabel("held-out failure (mean over 6 worlds, 12 seeds)")
    ax.invert_xaxis()
    ax.set_title("Repair vs replay start time (95% CI over seeds)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for src in ("real", "match", "sim"):
        ax.plot(
            x,
            [r["nominal_means"][f"{src}_{k}:nominal_success"] for k in FIXED],
            "o-",
            color=COLORS[src],
            label=LABELS[src],
        )
    ax.axhline(r["nominal_means"]["A:nominal_success"], color="k", ls=":", label="A no repair")
    ax.axhline(
        r["nominal_means"]["A:nominal_success"] - 0.02,
        color="r",
        ls="--",
        lw=1,
        label="-2 pp margin",
    )
    ax.axhline(r["nominal_means"]["B:nominal_success"], color="k", ls="--", lw=1, label="B")
    ax.invert_xaxis()
    ax.set_title("Nominal success")
    ax.set_xlabel("seconds before onset")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    for w in worlds:
        c = r["setup"][w]["curve_real"]
        ax.plot(
            [p["lead_s"] for p in c], [p["success"] for p in c], color=COLORS["real"], alpha=0.6
        )
        c = r["setup"][w]["curve_sim"]
        ax.plot([p["lead_s"] for p in c], [p["success"] for p in c], color=COLORS["sim"], alpha=0.4)
    ax.invert_xaxis()
    ax.set_title(
        "Recoverability R(t), baseline policy\n(red real, green sim; one line per failure)"
    )
    ax.set_xlabel("seconds before onset")
    ax.set_ylabel("P(traverse | restart here), replay band")

    diag_panels = (
        (
            "survival",
            "survival horizon from restart (s)",
            lambda a: a["recoverability_band"]["survival_s"],
        ),
        (
            "immediate",
            "fraction failing within 0.1 s",
            lambda a: a["recoverability_band"]["immediate_termination"],
        ),
        (
            "signal",
            "ES gradient norm at baseline (arm's training mix)",
            lambda a: a["es_grad_norm"],
        ),
    )
    for ax, (_, title, fn) in zip(axes[1], diag_panels):
        for src in ("real", "match", "sim"):
            vals = [
                np.mean([fn(r["setup"][w]["diagnostics"][f"{src}_{k}"]) for w in worlds])
                for k in FIXED
            ]
            ax.plot(x, vals, "o-", color=COLORS[src], label=LABELS[src])
        ax.invert_xaxis()
        ax.set_title(title)
        ax.set_xlabel("seconds before onset")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Precursor time sweep, stage 1 (CPU slip cart-pole, toy_mechanism evidence)")
    fig.tight_layout()
    fig.savefig(out / "stage1_time_sweep.png", dpi=130)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, w in zip(axes.ravel(), worlds):
        pw = r["per_world_held_out_failure"][w]
        for src in ("real", "match", "sim"):
            ax.plot(x, [pw[f"{src}_{k}"] for k in FIXED], "o-", color=COLORS[src])
        ax.axhline(pw["B"], color="k", ls="--", lw=1)
        ax.axhline(pw["ADR"], color="k", ls="-.", lw=1)
        ax.invert_xaxis()
        ax.set_title(f"{w} {r['worlds'][w]}")
    fig.suptitle(
        "Per-world held-out failure (red real, blue matched success, green sim; "
        "B dashed, ADR dash-dot)"
    )
    fig.tight_layout()
    fig.savefig(out / "stage1_per_world.png", dpi=110)
    print(out)


def stage2_plot(path: str) -> None:
    d = Path(path) / "stage2"
    r = json.loads((d / "stage2.json").read_text())
    p = r["P_star"]
    order = [
        "A",
        "ADR",
        "B",
        "real_T0",
        "real_Rsel",
        "real_entry",
        "C",
        f"match_{p}",
        f"sim_{p}",
        f"real_{p}",
    ]
    sm = r["seed_means"]
    m, e = zip(*(_ci(list(sm[a].values())) for a in order))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cols = ["#999"] * 3 + ["#b5471b"] * 3 + ["#555", "#3a6ea5", "#5a8a3a", "#b5471b"]
    ax.bar(range(len(order)), m, yerr=e, color=cols, capsize=3)
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
    ax.set_ylabel("held-out failure, hidden worlds W7-W12")
    ax.set_title(f"Stage 2 (hidden worlds, new seeds): GO = {r['GO']}")
    fig.tight_layout()
    (d / "figures").mkdir(exist_ok=True)
    fig.savefig(d / "figures" / "stage2_arms.png", dpi=130)
    print(d / "figures")


if __name__ == "__main__":
    main(sys.argv[1])
    if (Path(sys.argv[1]) / "stage2" / "stage2.json").exists():
        stage2_plot(sys.argv[1])
