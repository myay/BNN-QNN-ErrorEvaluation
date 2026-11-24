"""Run the accuracy-over-bit-error-rate experiments of Fig. 3 of the MCEL paper.

  M. Yayla, A. Kumar: "MCEL: Margin-Based Cross-Entropy Loss for Error-Tolerant
  Quantized Neural Networks", arXiv:2603.05048, 2026.

Every case of Fig. 3 (2-, 4-, 8-bit QNNs and BNNs on FashionMNIST/VGG3,
SVHN/VGG7, CIFAR10/MobileNetV2, Imagenette/ResNet18) is trained once per loss:
  cel   standard cross-entropy (the paper's baseline, "CEL")
  mcel  margin-based cross-entropy with margin m (the paper's method, "MCEL")
  mhl   modified hinge loss (BNN baseline of the paper)
and then evaluated under bit flips injected into the weights (inference only,
10 repetitions per error rate). Each run is one call of run_master.py.

Settings and where they come from:
  * hyperparameters: Table II of the paper (Adam in all cases)
  * error-rate grids: the grids in the former run_*.py scripts for that model and bit width
    (8-bit grids are not documented anywhere; the 4-bit grid of the same model is used)
  * MHL hinge parameter b: the value in the former BNN scripts (VGG3: 512, VGG7: 128);
    not stated in the paper
  * margins: the paper evaluates powers of two up to 128 and m=192 (m=256 for the
    FashionMNIST BNN); the default here is 128 and 192 (+256 for the FashionMNIST BNN).
    Use --margins to change it.
  * no bit flips during training (as in the paper)

Usage (from the repository root):
  python3 scripts/run_paper_fig3.py --print-only            # list all commands
  python3 scripts/run_paper_fig3.py --profile smoke         # every case runs (1 batch, ~30 min)
  python3 scripts/run_paper_fig3.py --profile quick --only qnn4_   # sanity check, 10 epochs (~1 h)
  python3 scripts/run_paper_fig3.py                         # full experiments (days on one GPU)
  python3 scripts/run_paper_fig3.py --only qnn4_fmnist      # a subset (substring of the run name)
  python3 scripts/plot_error_tolerance.py paper_runs        # figure + tables

Finished runs are skipped when the script is started again (delete a run's
DONE file or use --force to repeat it). Each run writes its log to
<out-dir>/<run>/log.txt and its results to <out-dir>/<run>/experiments/.../results.jsonl.
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_MASTER = os.path.join(ROOT, "run_master.py")


def grid(divisor, count):
    """[j/divisor for j in range(count)], as written in the former scripts."""
    return [j / divisor for j in range(count)]


# Table II of the paper: epochs, batch size, learning rate, step size, gamma
def hp(epochs, batch_size, lr, step_size, gamma):
    return {"epochs": epochs, "batch_size": batch_size, "lr": lr, "step_size": step_size, "gamma": gamma}


HP_VGG3 = hp(100, 256, 0.001, 10, 0.5)
HP_VGG7_QNN = hp(50, 256, 0.001, 5, 0.25)
HP_VGG7_BNN = hp(100, 256, 0.001, 10, 0.5)
HP_MOBILENET = hp(100, 128, 0.001, 10, 0.5)
HP_RESNET18 = hp(100, 128, 0.001, 10, 0.5)

# column of Fig. 3: (short name, model, dataset)
FMNIST = ("fmnist", "vgg3", "FMNIST")
SVHN = ("svhn", "vgg7", "SVHN")
CIFAR10 = ("cifar10", "mobilenetv2", "CIFAR10")
IMAGENETTE = ("imagenette", "resnet18", "IMAGENETTE")

# (row, column, bits, hyperparameters, error-rate grid, MHL parameter b)
# bits=None means BNN.
CASES = [
    ("qnn2", FMNIST, 2, HP_VGG3, grid(100, 31), None),
    ("qnn4", FMNIST, 4, HP_VGG3, grid(100, 31), None),
    ("qnn8", FMNIST, 8, HP_VGG3, grid(100, 31), None),
    ("qnn2", SVHN, 2, HP_VGG7_QNN, grid(8000, 41), None),
    ("qnn4", SVHN, 4, HP_VGG7_QNN, grid(2000, 41), None),
    ("qnn8", SVHN, 8, HP_VGG7_QNN, grid(2000, 41), None),
    ("qnn2", CIFAR10, 2, HP_MOBILENET, grid(16000, 61), None),
    ("qnn4", CIFAR10, 4, HP_MOBILENET, grid(8000, 61), None),
    ("qnn8", CIFAR10, 8, HP_MOBILENET, grid(8000, 61), None),
    ("qnn2", IMAGENETTE, 2, HP_RESNET18, grid(8000, 41), None),
    ("qnn4", IMAGENETTE, 4, HP_RESNET18, grid(2000, 41), None),
    ("qnn8", IMAGENETTE, 8, HP_RESNET18, grid(2000, 41), None),
    ("bnn", FMNIST, None, HP_VGG3, grid(100, 31), 512),
    ("bnn", SVHN, None, HP_VGG7_BNN, grid(200, 41), 128),
]

DEFAULT_MARGINS = (128, 192)
EXTRA_MARGINS = {("bnn", "fmnist"): (256,)}  # the paper: m=256 helps only for the FashionMNIST BNN
REPETITIONS = 10
QUICK_EPOCHS = 10
QUICK_REPETITIONS = 3


class Run(object):
    def __init__(self, name, args, est_minutes):
        self.name = name
        self.args = args
        self.est_minutes = est_minutes


# rough minutes per epoch / per evaluation of the test set on a GTX 1060 (measured with run_master.py)
EPOCH_MIN = {"vgg3": 0.13, "vgg7": 0.5, "mobilenetv2": 0.55, "resnet18": 0.85}
TEST_MIN = {"vgg3": 0.016, "vgg7": 0.07, "mobilenetv2": 0.04, "resnet18": 0.12}


def build_runs(profile, margins):
    runs = []
    for row, (col, model, dataset), bits, hparams, rates, mhl_b in CASES:
        losses = [("cel", "cel", None)]
        ms = list(margins) + [m for m in EXTRA_MARGINS.get((row, col), ()) if m not in margins]
        if profile in ("smoke", "quick"):
            ms = ms[:1]
        losses += [("mcel_m{:g}".format(m), "celm", m) for m in ms]
        if mhl_b is not None:
            losses.append(("mhl_b{:g}".format(mhl_b), "mhl", mhl_b))

        n = len(rates)
        if profile == "smoke":
            rates = [rates[0], rates[n // 4], rates[n // 2], rates[-1]]
        elif profile == "quick":
            rates = [rates[0], rates[n // 8], rates[n // 4], rates[n // 2], rates[-1]]
        for loss_name, criterion, param in losses:
            name = "{}_{}_{}__{}".format(row, col, model, loss_name)
            a = ["--model", model, "--type", "binary" if bits is None else "quantized", "--dataset", dataset,
                 "--optimizer", "adam", "--criterion", criterion,
                 "--batch-size", str(hparams["batch_size"]), "--lr", str(hparams["lr"]),
                 "--step-size", str(hparams["step_size"]), "--gamma", str(hparams["gamma"]),
                 "--error-rates", ",".join(repr(r) for r in rates)]
            if bits is not None:
                a += ["--bits", str(bits)]
            if param is not None:
                a += ["--margin-param", "{:g}".format(param)]
            if profile == "smoke":
                a += ["--epochs", "1", "--dry-run", "--test-errors", "2"]
                est = 0.5 + 8 * TEST_MIN[model]
            elif profile == "quick":
                a += ["--epochs", str(QUICK_EPOCHS), "--test-errors", str(QUICK_REPETITIONS)]
                est = 0.5 + QUICK_EPOCHS * (EPOCH_MIN[model] + TEST_MIN[model]) \
                    + len(rates) * QUICK_REPETITIONS * 1.3 * TEST_MIN[model]
            else:
                a += ["--epochs", str(hparams["epochs"]), "--test-errors", str(REPETITIONS)]
                est = hparams["epochs"] * (EPOCH_MIN[model] + TEST_MIN[model]) \
                    + len(rates) * REPETITIONS * 1.3 * TEST_MIN[model]
            runs.append(Run(name, a, est))
    return runs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the Fig. 3 experiments of the MCEL paper with run_master.py.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--profile", choices=("paper", "quick", "smoke"), default="paper",
                   help="'paper': full training and 10 repetitions per error rate; "
                        "'quick': 10 epochs, 5 error rates, 3 repetitions, one margin - sanity check that the "
                        "networks learn and lose accuracy under bit errors; "
                        "'smoke': 1 training batch, 4 error rates, 2 repetitions, one margin - checks that "
                        "every case runs")
    p.add_argument("--margins", type=str, default=",".join(str(m) for m in DEFAULT_MARGINS),
                   help="comma-separated MCEL margins m")
    p.add_argument("--only", type=str, nargs="*", default=None,
                   help="run only runs whose name contains one of these substrings (e.g. qnn4_fmnist, __cel, bnn)")
    p.add_argument("--out-dir", type=str, default="paper_runs", help="output root directory")
    p.add_argument("--data-dir", type=str, default="data", help="dataset root directory (passed to run_master.py)")
    p.add_argument("--gpu-num", type=int, default=0, help="GPU index")
    p.add_argument("--seed", type=int, default=1, help="random seed for training")
    p.add_argument("--deterministic", action="store_true", help="pass --deterministic to run_master.py")
    p.add_argument("--force", action="store_true", help="repeat runs that are already DONE")
    p.add_argument("--print-only", action="store_true", help="only print the commands and the time estimate")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        margins = [float(m) for m in args.margins.split(",") if m.strip()]
    except ValueError:
        sys.exit("--margins must be a comma-separated list of numbers")
    if any(m <= 0 for m in margins):
        sys.exit("--margins must be > 0 (use the 'cel' runs for the standard cross-entropy)")

    runs = build_runs(args.profile, margins)
    if args.only:
        runs = [r for r in runs if any(s in r.name for s in args.only)]
    if not runs:
        sys.exit("no run matches --only {}".format(args.only))

    out_root = os.path.abspath(args.out_dir)
    common = ["--data-dir", os.path.abspath(args.data_dir), "--gpu-num", str(args.gpu_num),
              "--seed", str(args.seed), "--verbosity", "min"]
    if args.deterministic:
        common.append("--deterministic")

    total = sum(r.est_minutes for r in runs)
    print("{} runs, profile '{}', estimated {:.1f} h on a GTX 1060 (rough)".format(
        len(runs), args.profile, total / 60.0))
    if args.print_only:
        for r in runs:
            out = os.path.join(out_root, r.name)
            print("python3 run_master.py {} {} --output-dir {}  # ~{:.0f} min".format(
                " ".join(r.args), " ".join(common), out, r.est_minutes))
        return 0

    status = []
    for i, r in enumerate(runs, 1):
        out = os.path.join(out_root, r.name)
        done = os.path.join(out, "DONE")
        if os.path.exists(done) and not args.force:
            print("[{}/{}] {:<40} already done, skipped".format(i, len(runs), r.name))
            status.append((r.name, "skipped", 0.0))
            continue
        if os.path.exists(done):
            os.remove(done)
        if not os.path.isdir(out):
            os.makedirs(out)
        cmd = [sys.executable, RUN_MASTER] + r.args + common + ["--output-dir", out]
        print("[{}/{}] {:<40} running (~{:.0f} min) ...".format(i, len(runs), r.name, r.est_minutes))
        sys.stdout.flush()
        t0 = time.time()
        with open(os.path.join(out, "log.txt"), "w") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        minutes = (time.time() - t0) / 60.0
        if rc == 0:
            open(done, "w").close()
            status.append((r.name, "ok", minutes))
        else:
            status.append((r.name, "FAILED (exit {}, see {})".format(rc, os.path.join(out, "log.txt")), minutes))
        print("    {} after {:.1f} min".format(status[-1][1], minutes))

    print("\nsummary:")
    for name, st, minutes in status:
        print("  {:<40} {:<10} {:6.1f} min".format(name, st, minutes))
    failed = [s for s in status if s[1].startswith("FAILED")]
    print("{} ok, {} skipped, {} failed".format(sum(s[1] == "ok" for s in status),
                                                sum(s[1] == "skipped" for s in status), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
