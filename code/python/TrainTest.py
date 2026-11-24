"""Training / evaluation / bit-error sweep shared by run_master.py.

Single copy of the train_m / test_m / test_m_error methods that were
copy-pasted into every model class of the run_*.py scripts, plus the margin
statistics plot and the tikz-style result printing. Numerics are unchanged.
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless servers
import matplotlib.pyplot as plt  # noqa: E402

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation  # noqa: E402
from Utils import set_layer_mode  # noqa: E402

QUANTIZED_LAYERS = (QuantizedActivation, QuantizedLinear, QuantizedConv2d)


def train_epoch(model, criterion, device, train_loader, optimizer, epoch, log_interval=10,
                verbosity="all", dry_run=False, max_batches=None, margins_storage=None):
    """One training epoch (== train_m). Appends the epoch's mean top1-top2 logit margin to margins_storage.

    Returns the loss of the last processed batch.
    """
    model.train()
    set_layer_mode(model, "train")  # propagate information about training to all layers
    epoch_margins = []
    loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # statistics on "highest logit - second highest logit"
        top2 = torch.topk(output, k=2, dim=1).values
        epoch_margins.append(top2[:, 0] - top2[:, 1])
        loss = criterion.applyCriterion(output, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if verbosity == "all":
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break
    if margins_storage is not None and epoch_margins:
        margins_storage.append(torch.cat([t.flatten() for t in epoch_margins]).mean())
    return None if loss is None else loss.item()


def test(model, criterion, device, test_loader, verbosity="all"):
    """Accuracy in percent on test_loader (== test_m)."""
    model.eval()
    set_layer_mode(model, "eval")  # propagate information about eval to all layers
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion.applyCriterion(output, target).mean()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if verbosity == "all":
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * (correct / len(test_loader.dataset))


def set_error_rate(model, perror):
    """Set the bit error rate on every quantized layer that has an error model."""
    for layer in model.children():
        if isinstance(layer, QUANTIZED_LAYERS) and layer.error_model is not None:
            layer.error_model.updateErrorModel(perror)


def reset_error_rate(model):
    for layer in model.children():
        if isinstance(layer, QUANTIZED_LAYERS) and layer.error_model is not None:
            layer.error_model.resetErrorModel()


def test_error(model, criterion, device, test_loader, perror, verbosity="all"):
    """Accuracy with bit error rate `perror` injected (== test_m_error); error model is reset afterwards."""
    model.eval()
    set_layer_mode(model, "eval")
    set_error_rate(model, perror)
    if verbosity == "all":
        print("Error rate: ", perror)
    accuracy = test(model, criterion, device, test_loader, verbosity)
    reset_error_rate(model)
    return accuracy


def error_sweep(model, criterion, device, test_loader, perrors, repetitions, verbosity="all"):
    """Run `repetitions` evaluations per error rate. Returns list of accuracy lists (one per rate)."""
    c_list = []
    for perror in perrors:
        if verbosity == "all":
            print(perror)
        acc_list = []
        for _ in range(repetitions):
            acc_list.append(test_error(model, criterion, device, test_loader, perror, verbosity))
        c_list.append(acc_list)
    return c_list


def print_tikz_data(in_array, nr):
    """Prints '<rate in %> <mean> <max-mean> <mean-min>' (same format as the originals)."""
    accs_mean = np.mean(np.array(in_array), axis=0)
    accs_min = np.min(np.array(in_array), axis=0)
    accs_max = np.max(np.array(in_array), axis=0)
    print("{} {} {} {}".format(nr, accs_mean, accs_max - accs_mean, accs_mean - accs_min))


def save_margin_histogram(margin_list, bucket_size=10, out_path="margin_histogram.png"):
    """Histogram of all margins (unused by default in the originals, kept for completeness)."""
    margins = torch.cat([m.flatten().detach().cpu() for m in margin_list]).numpy()
    bins = np.arange(margins.min(), margins.max() + bucket_size, bucket_size)
    plt.figure()
    plt.hist(margins, bins=bins)
    plt.xlabel("Logit margin")
    plt.ylabel("Absolute frequency")
    plt.title("Margin distribution")
    plt.grid(True)
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_margin_over_batches(margin_list, out_path="margins_vs_batches.png", statistic="mean",
                             txt_path="output-margins.txt"):
    """Plot + text file of the per-epoch margin statistic (same files as the originals)."""
    values = []
    for margins in margin_list:
        m = margins.detach().cpu().flatten().numpy()
        if statistic == "mean":
            values.append(np.mean(m))
        elif statistic == "median":
            values.append(np.median(m))
        else:
            raise ValueError("statistic must be 'mean' or 'median'")
    batches = np.arange(len(values))
    for p in (out_path, txt_path):
        if os.path.dirname(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savetxt(txt_path, np.array(values))
    print("len vals", len(values))
    plt.figure()
    plt.plot(batches, values)
    plt.xlabel("Batch")
    plt.ylabel("{} logit margin".format(statistic.capitalize()))
    plt.title("Margin development over batches")
    plt.grid(True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
