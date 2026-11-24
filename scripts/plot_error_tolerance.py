"""Plot accuracy over bit error rate from run_master.py results (Fig. 3 style of the MCEL paper).

Reads every results.jsonl below the given directories (written by run_master.py,
e.g. via scripts/run_paper_fig3.py) and writes to --out:
  accuracy_over_ber.png / .pdf  one panel per (bit width, model/dataset):
                                rows 2-bit / 4-bit / 8-bit QNN / BNN, columns model + dataset,
                                one curve per loss (CEL, MCEL m=..., MHL b=...),
                                mean over the repetitions with min/max error bars
  tikz/<panel>__<loss>.dat      '<BER in %> <mean> <max-mean> <mean-min>' per line, the format
                                the run scripts printed for pgfplots (error bars: plus/minus)
  summary.md                    accuracy without errors and at selected error rates per panel

Several results for the same configuration (e.g. different seeds) are pooled:
mean/min/max are taken over all their repetitions.

Usage:  python3 scripts/plot_error_tolerance.py paper_runs [more dirs] [--out paper_runs/plots]
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATASET_LABEL = {"FMNIST": "FashionMNIST", "SVHN": "SVHN", "CIFAR10": "CIFAR10", "CIFAR100": "CIFAR100",
                 "IMAGENETTE": "Imagenette", "MNIST": "MNIST", "KMNIST": "KMNIST"}
MODEL_LABEL = {"vgg3": "VGG3", "vgg7": "VGG7", "vgg7_wide": "VGG7 (wide)", "fc": "FC", "resnet18": "ResNet18",
               "mobilenetv2": "MobileNetV2", "mobilenetv2_2x": "MobileNetV2 (2x head)"}
COLUMN_ORDER = [("vgg3", "FMNIST"), ("vgg7", "SVHN"), ("mobilenetv2", "CIFAR10"), ("resnet18", "IMAGENETTE")]


def load_results(dirs):
    """List of result dicts from all results.jsonl files below dirs."""
    results = []
    for d in dirs:
        for root, _, files in os.walk(d):
            if "results.jsonl" not in files:
                continue
            path = os.path.join(root, "results.jsonl")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        print("warning: skipping malformed line in", path)
                        continue
                    if r.get("test_error"):
                        r["_path"] = path
                        results.append(r)
    return results


def row_key(r):
    return ("BNN", 99) if r["type"] == "binary" else ("{}-bit QNN".format(r["bits"]), r["bits"])


def loss_label(r):
    crit = r.get("traincrit", "")
    if crit.startswith("CEL"):
        return "CEL", (0, 0)
    if crit.startswith("celm"):
        return "MCEL m={:g}".format(r["margin_param"]), (1, r["margin_param"])
    if crit.startswith("MHL"):
        return "MHL b={:g}".format(r["margin_param"]), (2, r["margin_param"])
    return crit, (3, 0)


def aggregate(results):
    """{(row, col): {loss_label: {perror: [accuracies]}}}, with ordering info."""
    panels = {}
    row_sort, loss_sort = {}, {}
    for r in results:
        row, rs = row_key(r)
        row_sort[row] = rs
        col = (r["arch"], r["dataset"])
        label, ls = loss_label(r)
        loss_sort[label] = ls
        curves = panels.setdefault((row, col), {})
        points = curves.setdefault(label, {})
        for entry in r["test_error"]:
            points.setdefault(round(entry["perror"], 10), []).extend(entry["accuracies"])
    rows = sorted(row_sort, key=lambda k: row_sort[k])
    cols_present = {c for (_, c) in panels}
    cols = [c for c in COLUMN_ORDER if c in cols_present] + sorted(cols_present - set(COLUMN_ORDER))
    return panels, rows, cols, loss_sort


def curve_arrays(points):
    rates = sorted(points)
    acc = [np.array(points[p], dtype=float) for p in rates]
    x = np.array(rates) * 100.0
    mean = np.array([a.mean() for a in acc])
    lo = np.array([a.min() for a in acc])
    hi = np.array([a.max() for a in acc])
    n = min(len(a) for a in acc)
    return x, mean, lo, hi, n


def curve_style(label, loss_sort, labels_in_panel):
    kind = loss_sort[label][0]
    if kind == 0:
        return {"color": "black", "marker": "s"}
    if kind == 2:
        return {"color": "tab:orange", "marker": "^"}
    # MCEL: shades of blue (the paper draws MCEL in blue), darker = larger margin
    mcel = sorted([l for l in labels_in_panel if loss_sort[l][0] == 1], key=lambda l: loss_sort[l])
    shades = plt.cm.Blues(np.linspace(0.5, 0.95, max(len(mcel), 1)))
    return {"color": shades[mcel.index(label)], "marker": "o"}


def panel_title(col):
    arch, dataset = col
    return "{} / {}".format(MODEL_LABEL.get(arch, arch), DATASET_LABEL.get(dataset, dataset))


def file_tag(row, col):
    return "{}_{}_{}".format(row.replace(" ", "").replace("-", ""), col[0], col[1]).lower()


def plot(panels, rows, cols, loss_sort, out_dir):
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.2 * len(cols), 3.2 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i][j]
            curves = panels.get((row, col))
            if not curves:
                ax.axis("off")
                continue
            labels = sorted(curves, key=lambda l: loss_sort[l])
            for label in labels:
                x, mean, lo, hi, n = curve_arrays(curves[label])
                st = curve_style(label, loss_sort, labels)
                ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], label=label, color=st["color"],
                            marker=st["marker"], markersize=3, linewidth=1.2, capsize=2, elinewidth=0.8)
            ax.set_title("{}, {}".format(row, panel_title(col)), fontsize=10)
            ax.set_xlabel("Bit error rate (%)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, "accuracy_over_ber." + ext)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", path)
    plt.close(fig)


def write_tikz(panels, out_dir):
    tdir = os.path.join(out_dir, "tikz")
    if not os.path.isdir(tdir):
        os.makedirs(tdir)
    count = 0
    for (row, col), curves in sorted(panels.items()):
        for label, points in curves.items():
            x, mean, lo, hi, _ = curve_arrays(points)
            name = "{}__{}.dat".format(file_tag(row, col), label.lower().replace(" ", "_").replace("=", ""))
            with open(os.path.join(tdir, name), "w") as f:
                for xi, m, l, h in zip(x, mean, lo, hi):
                    f.write("{} {} {} {}\n".format(xi, m, h - m, m - l))
            count += 1
    print("wrote {} tikz data files to {}".format(count, tdir))


def write_summary(panels, rows, cols, loss_sort, out_dir):
    lines = ["# Accuracy over bit error rate", "",
             "Mean accuracy in % over all repetitions (min-max in brackets).", ""]
    for row in rows:
        for col in cols:
            curves = panels.get((row, col))
            if not curves:
                continue
            labels = sorted(curves, key=lambda l: loss_sort[l])
            all_rates = sorted({p for l in labels for p in curves[l]})
            # error-free, a quarter, half and the end of the grid
            picks = sorted({all_rates[0], all_rates[len(all_rates) // 4], all_rates[len(all_rates) // 2],
                            all_rates[-1]})
            lines.append("## {}, {}".format(row, panel_title(col)))
            lines.append("")
            lines.append("| loss | reps | " + " | ".join("BER {:g}%".format(p * 100) for p in picks) + " |")
            lines.append("|---|---|" + "---|" * len(picks))
            for label in labels:
                pts = curves[label]
                cells = []
                for p in picks:
                    if p in pts:
                        a = np.array(pts[p], dtype=float)
                        cells.append("{:.2f} ({:.2f}-{:.2f})".format(a.mean(), a.min(), a.max()))
                    else:
                        cells.append("-")
                n = min(len(v) for v in pts.values())
                lines.append("| {} | {} | {} |".format(label, n, " | ".join(cells)))
            lines.append("")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot accuracy over bit error rate from run_master.py results.")
    p.add_argument("dirs", nargs="+", help="directories to search for results.jsonl")
    p.add_argument("--out", type=str, default=None, help="output directory (default: <first dir>/plots)")
    args = p.parse_args(argv)
    out_dir = args.out or os.path.join(args.dirs[0], "plots")
    results = load_results(args.dirs)
    if not results:
        sys.exit("no results.jsonl with an error sweep found below {}".format(", ".join(args.dirs)))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    panels, rows, cols, loss_sort = aggregate(results)
    print("{} results, {} panels".format(len(results), len(panels)))
    plot(panels, rows, cols, loss_sort, out_dir)
    write_tikz(panels, out_dir)
    write_summary(panels, rows, cols, loss_sort, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
