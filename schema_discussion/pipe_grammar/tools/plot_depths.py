#!/usr/bin/env python3
"""Plot observed pipe-block depth distributions from the frozen census."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]
ORDER = [
    ("template_fa4_prefill", "Template FA4 prefill"),
    ("template_fa4_decode", "Template FA4 decode"),
    ("template_gdn_prefill", "Template GDN"),
    ("template_qsa_prefill", "Template QSA"),
    ("extract_gdn", "Extract GDN"),
    ("extract_gemm", "Extract GEMM"),
    ("extract_mla", "Extract MLA"),
    ("extract_sparse", "Extract Sparse"),
    ("extract_fmha", "Extract FMHA"),
    ("extract_moe", "Extract MoE"),
]


def main():
    data = json.loads((BASE / "data/summary.json").read_text())
    colors = ["#d6dce5", "#93b4cf", "#447fa7", "#235676", "#122f43"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), sharey=True)
    levels = range(1, 6)
    for ax, field, title in zip(axes, ["root_counts", "counts"], ["Root YAML only", "All authored YAML nodes"]):
        rows = [data["operators"][op][field]["B_local"] for op, _ in ORDER]
        totals = np.array([row["pipe_blocks"] for row in rows])
        depths = np.array([[row["depth_counts"].get(str(level), 0) for level in levels] for row in rows])
        assert np.array_equal(depths.sum(axis=1), totals)
        percentages = depths * 100 / totals[:, None]
        left = np.zeros(len(ORDER))
        for index, level in enumerate(levels):
            ax.barh(np.arange(len(ORDER)), percentages[:, index], left=left,
                    color=colors[index], edgecolor="white", linewidth=.6,
                    label="Depth " + str(level), height=.7)
            left += percentages[:, index]
        for index, total in enumerate(totals):
            ax.text(102, index, f"{total:,}", va="center", fontsize=9)
        ax.text(102, -.85, "Blocks", fontsize=9, color="#4c5564")
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=20)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e6e9ee", linewidth=.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(ORDER)), [label for _, label in ORDER], fontsize=10)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.55, .04))
    fig.suptitle("Observed nesting of pipe blocks", x=.19, y=.98, ha="left", fontsize=15, weight="bold")
    fig.text(.19, .925, "Depth 1 = role's top-level pipe. Local YAML aliases expanded; runtime loops not expanded.",
             fontsize=9, color="#4c5564")
    fig.text(.19, .025, "Counts describe syntax, not execution time, redundancy, or semantic equivalence.",
             fontsize=9, color="#4c5564")
    fig.subplots_adjust(left=.19, right=.99, top=.82, bottom=.17, wspace=.13)
    target = BASE / "figures"
    target.mkdir(exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(target / ("pipe_depths." + extension), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figures/pipe_depths.png and .svg")


if __name__ == "__main__":
    main()
