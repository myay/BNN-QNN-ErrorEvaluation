"""Master script for binarized (BNN) and quantized (QNN) training + bit-error evaluation.

Replaces the former per-model run_*.py scripts and run.py.
Models live in code/python/models.py, datasets in code/python/Datasets.py, the
train/test/error-sweep loops in code/python/TrainTest.py.

Examples (see --help for all options):
  python3 run_master.py --model=vgg3 --type=binary --epochs=50 --test-errors=10
  python3 run_master.py --model=vgg7 --type=quantized --bits=4 --dataset=SVHN
  python3 run_master.py --model=resnet18 --type=quantized --bits=2 --margin-param=96
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code", "python"))

from Utils import Clippy, Criterion, binary_hingeloss, margin_softmax_ce  # noqa: E402
from QuantMethods import VALID_BITS  # noqa: E402
import models  # noqa: E402
from models import (get_model, get_spec, resolve_bits, default_save_name,  # noqa: E402
                    MODEL_REGISTRY, MODEL_NAMES, KINDS)
from Datasets import DATASET_NAMES, canonical_dataset, dataset_geometry, get_datasets, get_loaders  # noqa: E402
import TrainTest  # noqa: E402

torch.set_printoptions(sci_mode=False)

CRITERIA = ("celm", "cel", "mhl")
OPTIMIZERS = ("adam", "clippy")
HP_KEYS = ("batch_size", "epochs", "lr", "step_size", "gamma")


def _model_defaults_epilog():
    lines = ["model-specific defaults (used when the corresponding option is omitted):",
             "  {:<15} {:<10} {:<11} {:<5} {:<7} {:>5} {:>6} {:>8} {:>5} {:>6}  {}".format(
                 "model", "type", "dataset", "bits", "optim", "bs", "epochs", "lr", "step", "gamma",
                 "error-rate grid")]
    for name, kind in models.valid_combinations():
        s = get_spec(name, kind)
        hp = s.hparams
        rates = s.error_rates
        grid = "{} rates, {:g}..{:g} (step {:g})".format(len(rates), rates[0], rates[-1], rates[1] - rates[0])
        lines.append("  {:<15} {:<10} {:<11} {:<5} {:<7} {:>5} {:>6} {:>8g} {:>5} {:>6g}  {}".format(
            name, kind, s.default_dataset, str(s.default_bits or "-"), s.optimizer, hp["batch_size"],
            hp["epochs"], hp["lr"], hp["step_size"], hp["gamma"], grid))
    lines.append("")
    lines.append("valid bit widths for --type=quantized: " + ", ".join(str(b) for b in VALID_BITS))
    lines.append("--type=binary together with --bits is rejected with an error.")
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(
        description="Train and evaluate binarized (BNN) / quantized (QNN) networks under bit errors.",
        epilog=_model_defaults_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("model")
    g.add_argument("--model", type=str.lower, required=True, choices=MODEL_NAMES,
                   help="architecture (required). Choices: %(choices)s")
    g.add_argument("--type", dest="type", type=str.lower, default=None, choices=KINDS,
                   help="'binary' (BNN, {-1,+1}) or 'quantized' (QNN, uniform n-bit). "
                        "Default: the only type the model has; required if it has both (vgg3, vgg7)")
    g.add_argument("--bits", type=int, default=None,
                   help="bit width of a quantized model ({}). Default: model-specific "
                        "(vgg3: 4, vgg7: 8, resnet18/mobilenetv2*: 4). Not allowed with --type=binary "
                        "(error)".format(", ".join(str(b) for b in VALID_BITS)))
    g.add_argument("--load-model-path", type=str, default=None,
                   help="state_dict to load right after building the model (before training/testing). "
                        "Checkpoints saved by the former run scripts load as well (default: None)")
    g.add_argument("--train-model", type=int, default=1, choices=(0, 1),
                   help="1: train for --epochs epochs, 0: skip training (e.g. evaluate a loaded model) (default: 1)")

    g = p.add_argument_group("data")
    g.add_argument("--dataset", type=str, default=None,
                   help="dataset: {} (default: model-specific, see table below)".format(
                       "/".join(DATASET_NAMES)))
    g.add_argument("--data-dir", type=str, default="data", help="dataset root directory (default: data)")
    g.add_argument("--image-size", type=int, default=None,
                   help="ImageNette resize size (default: 64)")
    g.add_argument("--batch-size", type=int, default=None, metavar="N",
                   help="training batch size (default: model-specific)")
    g.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                   help="test batch size (default: 1000)")
    g.add_argument("--num-workers", type=int, default=1, help="DataLoader workers (default: 1)")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=None, metavar="N",
                   help="number of training epochs (default: model-specific)")
    g.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (default: model-specific)")
    g.add_argument("--step-size", type=int, default=None, metavar="M",
                   help="StepLR step size in epochs (default: model-specific)")
    g.add_argument("--gamma", type=float, default=None, metavar="M",
                   help="StepLR decay factor (default: model-specific)")
    g.add_argument("--optimizer", type=str.lower, default=None, choices=OPTIMIZERS,
                   help="'adam' or 'clippy' (Adam, as in Utils.Clippy) (default: model-specific: "
                        "clippy for vgg3/vgg7_wide/fc, adam otherwise)")
    g.add_argument("--criterion", type=str.lower, default="celm", choices=CRITERIA,
                   help="loss: 'celm' = margin softmax cross-entropy (Utils.margin_softmax_ce), "
                        "'cel' = plain cross-entropy, 'mhl' = binary hinge loss with "
                        "b=--margin-param, which must be > 0 (default: celm)")
    g.add_argument("--margin-param", type=float, default=0,
                   help="margin m of 'celm' / b of 'mhl'; ignored for 'cel' (default: 0)")
    g.add_argument("--train-error-rate", type=float, default=0.0,
                   help="bit error rate injected during training and the per-epoch test "
                        "(default: 0.0)")
    g.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    g.add_argument("--deterministic", action="store_true", default=False,
                   help="set cudnn.deterministic=True / cudnn.benchmark=False. Without it "
                        "QNN training is not reproducible run-to-run even with a fixed seed; the bit-flip kernels "
                        "are always seeded from the clock (default: off)")
    g.add_argument("--dry-run", action="store_true", default=False,
                   help="train only on the first batch of each epoch (default: off)")
    g.add_argument("--max-train-batches", type=int, default=None, metavar="N",
                   help="stop each training epoch after N batches (default: no limit)")
    g.add_argument("--log-interval", type=int, default=10, metavar="N",
                   help="batches between training-loss prints (default: 10)")

    g = p.add_argument_group("bit-error evaluation")
    g.add_argument("--test-errors", type=int, default=1, metavar="N",
                   help="after training, evaluate every error rate N times (0 = skip the sweep) (default: 1)")
    g.add_argument("--error-rates", type=str, default=None,
                   help="comma-separated bit error rates for the sweep, e.g. '0,0.001,0.01' "
                        "(default: model-specific grid, see table below)")

    g = p.add_argument_group("system / output")
    g.add_argument("--gpu-num", type=int, default=0, metavar="N", help="GPU index to use (default: 0)")
    g.add_argument("--no-cuda", action="store_true", default=False,
                   help="disable CUDA. Not supported: the binarization/quantization kernels are CUDA-only, "
                        "so this exits with an error (kept for CLI compatibility) (default: off)")
    g.add_argument("--output-dir", type=str, default=".",
                   help="directory for experiments/<model>/results-<time>/results.jsonl, "
                        "output-margins.txt, margins_vs_batches.png and the saved model (default: .)")
    g.add_argument("--save-model", action="store_true", default=False,
                   help="save the trained state_dict (default: off)")
    g.add_argument("--save-path", type=str, default=None,
                   help="file for --save-model (default: <output-dir>/<model>_<type>[_<bits>bit].pt, "
                        "e.g. vgg3_binary.pt / vgg7_quantized_8bit.pt)")
    g.add_argument("--verbosity", type=str.lower, default="all", choices=("all", "min"),
                   help="'all' prints per-batch/per-epoch details, 'min' only the summary (default: all)")
    return p


def resolve_args(parser, args):
    """Validate the model/type/bits combination and fill model-specific defaults."""
    kinds = [k for k in KINDS if k in MODEL_REGISTRY[args.model]]
    if args.type is None:
        if len(kinds) != 1:
            parser.error("--type is required for model '{}' (valid: {})".format(args.model, ", ".join(kinds)))
        args.type = kinds[0]
    try:
        spec = get_spec(args.model, args.type)
    except ValueError as e:
        parser.error(str(e))
    try:
        args.bits = resolve_bits(args.model, args.type, args.bits)
    except ValueError as e:
        parser.error(str(e))

    try:
        args.dataset = canonical_dataset(args.dataset or spec.default_dataset)
    except ValueError as e:
        parser.error(str(e))
    if args.image_size is not None and args.dataset != "IMAGENETTE":
        print("Warning: --image-size only applies to IMAGENETTE; ignored for {}".format(args.dataset))
        args.image_size = None

    for k in HP_KEYS:
        if getattr(args, k) is None:
            setattr(args, k, spec.hparams[k])
    if args.optimizer is None:
        args.optimizer = spec.optimizer

    if args.error_rates is None:
        args.error_rates = list(spec.error_rates)
    else:
        try:
            args.error_rates = [float(v) for v in args.error_rates.split(",") if v.strip() != ""]
        except ValueError:
            parser.error("--error-rates must be a comma-separated list of numbers")
        if not args.error_rates:
            parser.error("--error-rates is empty")
    for r in args.error_rates + [args.train_error_rate]:
        if not 0.0 <= r <= 1.0:
            parser.error("error rates must be in [0, 1], got {}".format(r))
    if args.criterion == "mhl" and args.margin_param <= 0:
        parser.error("--criterion=mhl requires --margin-param > 0 (it is the hinge margin b and the loss "
                     "is divided by it), got {:g}".format(args.margin_param))
    if args.test_errors < 0:
        parser.error("--test-errors must be >= 0")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be >= 1")
    if args.save_path is None:
        args.save_path = os.path.join(args.output_dir, default_save_name(args.model, args.type, args.bits))
    return spec


def make_criterion(name, param):
    if name == "celm":
        return Criterion(margin_softmax_ce, "celm_train", param=param)
    if name == "mhl":
        return Criterion(binary_hingeloss, "MHL_train", param=param)
    return Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_train")


def create_exp_file(output_dir, model_name):
    """<output_dir>/experiments/<model_name>/results-<timestamp>/results.jsonl (as Utils.create_exp_folder)."""
    exp_path = os.path.join(os.path.abspath(output_dir), "experiments", model_name,
                            "results-" + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))
    try:
        os.makedirs(exp_path, 0o755, exist_ok=False)
    except OSError:
        print("Creation of the directory %s failed" % exp_path)
    else:
        print("Successfully created the directory %s" % exp_path)
    if not os.path.isdir(exp_path):
        raise IOError("cannot create results directory {}".format(exp_path))
    path = os.path.join(exp_path, "results.jsonl")
    if not os.path.exists(path):
        open(path, "w").close()
    return path


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_args(parser, args)

    if args.no_cuda or not torch.cuda.is_available():
        parser.error("CUDA is required: the binarization/quantization/error-injection kernels only run on GPU "
                     "({}).".format("--no-cuda given" if args.no_cuda else "no CUDA device available"))
    if args.gpu_num >= torch.cuda.device_count():
        parser.error("--gpu-num={} but only {} GPU(s) available".format(args.gpu_num, torch.cuda.device_count()))

    # fail now, not after hours of training, if the outputs cannot be written
    out_dirs = [args.output_dir]
    if args.save_model and os.path.dirname(args.save_path):
        out_dirs.append(os.path.dirname(args.save_path))
    for d in out_dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            parser.error("cannot create output directory {}: {}".format(d, e))
        if not os.access(d, os.W_OK):
            parser.error("output directory {} is not writable".format(d))

    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(args.gpu_num)
    device = torch.device("cuda")

    # data
    train_set, test_set = get_datasets(args.dataset, args.data_dir, args.image_size)
    train_loader, test_loader = get_loaders(train_set, test_set, args.batch_size, args.test_batch_size,
                                            num_workers=args.num_workers)
    in_channels, input_size, num_classes = dataset_geometry(args.dataset, args.image_size)

    # model (same RNG consumption order as the former run scripts, so --seed gives the same init)
    model = get_model(args.model, args.type, args.bits, num_classes=num_classes,
                      in_channels=in_channels, input_size=input_size).to(device)
    criterion = make_criterion(args.criterion, None if args.criterion == "cel" else args.margin_param)

    print("Config: model={} type={} bits={} dataset={} optimizer={} batch_size={} epochs={} lr={} "
          "step_size={} gamma={} seed={}".format(args.model, args.type, args.bits, args.dataset, args.optimizer,
                                                  args.batch_size, args.epochs, args.lr, args.step_size,
                                                  args.gamma, args.seed))
    print("Model name", model.name)
    print("Error model", model.method)
    print("Criterion", criterion.name)
    print("Criterion param", criterion.param)

    to_dump_path = create_exp_file(args.output_dir, model.name)

    if args.load_model_path is not None:
        print("Loaded model: ", args.load_model_path)
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))

    if args.optimizer == "clippy":
        optimizer = Clippy(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    margins_storage = []
    epoch_accuracies = []
    if args.train_model:
        if args.train_error_rate > 0:
            TrainTest.set_error_rate(model, args.train_error_rate)
            print("Training with bit error rate", args.train_error_rate)
        for epoch in range(1, args.epochs + 1):
            torch.cuda.synchronize()
            criterion.epoch += 1
            since = int(round(time.time() * 1000))
            TrainTest.train_epoch(model, criterion, device, train_loader, optimizer, epoch,
                                  log_interval=args.log_interval, verbosity=args.verbosity,
                                  dry_run=args.dry_run, max_batches=args.max_train_batches,
                                  margins_storage=margins_storage)
            if args.verbosity == "all":
                print('Epoch training time elapsed: {}ms'.format(int(round(time.time() * 1000)) - since))
            since = int(round(time.time() * 1000))
            acc = TrainTest.test(model, criterion, device, test_loader, args.verbosity)
            epoch_accuracies.append(acc)
            if args.verbosity == "all":
                print('Test time elapsed: {}ms'.format(int(round(time.time() * 1000)) - since))
            scheduler.step()
        TrainTest.reset_error_rate(model)

        if margins_storage:
            TrainTest.save_margin_over_batches(
                margins_storage, out_path=os.path.join(args.output_dir, "margins_vs_batches.png"),
                statistic="mean", txt_path=os.path.join(args.output_dir, "output-margins.txt"))

    sweep = []
    if args.test_errors:
        c_list = TrainTest.error_sweep(model, criterion, device, test_loader, args.error_rates,
                                       args.test_errors, args.verbosity)
        for perror, cc in zip(args.error_rates, c_list):
            TrainTest.print_tikz_data(cc, perror * 100)
            sweep.append({"perror": perror, "accuracies": cc})

    result = {
        "model": model.name, "arch": args.model, "type": args.type, "bits": args.bits, "dataset": args.dataset,
        "batchsize": args.batch_size, "epochs": args.epochs if args.train_model else 0, "learning_rate": args.lr,
        "gamma": args.gamma, "stepsize": args.step_size, "optimizer": args.optimizer, "seed": args.seed,
        "traincrit": criterion.name, "margin_param": criterion.param, "train_error_rate": args.train_error_rate,
        "epoch_test_accuracy": epoch_accuracies, "test_error": sweep,
    }
    with open(to_dump_path, "a") as f:
        json.dump(result, f)
        f.write("\n")
    print("Successfully stored results in %s" % to_dump_path)

    if args.save_model:
        torch.save(model.state_dict(), args.save_path)
        print("Saved model to", args.save_path)
    return result


if __name__ == "__main__":
    main()
