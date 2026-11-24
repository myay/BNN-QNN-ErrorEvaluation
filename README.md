# BNN-QNN-ErrorEvaluation
A framework for error tolerance training and evaluation of Binarized and Quantized Neural Networks

## CUDA-based Binarization, Quantization, and Error Injection

First, install PyTorch. For fast binarization/quantization and error injection during training, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```

The kernels only run on a GPU, so training and evaluation need a CUDA device. The code runs with Python 3.6 and PyTorch 1.4/1.5; the Python packages are listed in `requirements.txt` (`pip3 install -r requirements.txt`, install them before building the kernels).

## Training and evaluation under bit errors

After successful installation of all kernels, train and evaluate networks with `run_master.py` from the repository root.
Choose the architecture with `--model`, binarized (BNN) or quantized (QNN) with `--type`, and the bit width
of quantized networks with `--bits`:

```
# BNN: VGG3 on FashionMNIST, then a bit-error sweep with 10 repetitions per error rate
python3 run_master.py --model=vgg3 --type=binary --epochs=50 --test-errors=10

# QNN: VGG7 on SVHN with 8-bit weights, trained with the margin-based cross-entropy (m=128), model saved
python3 run_master.py --model=vgg7 --type=quantized --bits=8 --margin-param=128 --save-model

# evaluate the saved model under bit errors without training
python3 run_master.py --model=vgg7 --type=quantized --bits=8 --train-model=0 \
    --load-model-path=vgg7_quantized_8bit.pt --test-errors=10
```

Models: `vgg3`, `vgg7` (binary or quantized), `vgg7_wide`, `fc` (binary), `resnet18`, `mobilenetv2`,
`mobilenetv2_2x` (quantized, 2 to 8 bits). Datasets: MNIST, FMNIST, KMNIST, SVHN, CIFAR10, CIFAR100 (downloaded
to `--data-dir`, default `data`) and Imagenette (expected in `data/imagenette2/{train,val}`).

Options that are not given (dataset, bit width, batch size, epochs, learning rate, optimizer, error-rate grid, ...)
take model-specific defaults. `python3 run_master.py --help` lists all options and prints a table of these defaults.
Losses (`--criterion`): `celm` (default, margin-based cross-entropy, margin `--margin-param`), `cel` (standard
cross-entropy) and `mhl` (modified hinge loss, `--margin-param` > 0).

Outputs go to `--output-dir` (default: the current directory):
* `experiments/<model>/results-<time>/results.jsonl`: settings, test accuracy per epoch, and the accuracies of the
  bit-error sweep
* the sweep as lines `<bit error rate in %> <mean> <max-mean> <mean-min>` on the console (pgfplots error bars)
* `output-margins.txt` / `margins_vs_batches.png`: mean logit margin (top-1 minus top-2) per epoch
* the trained model with `--save-model` (default name `<model>_<type>[_<bits>bit].pt`)

Bit flips are injected only during evaluation, unless `--train-error-rate` is given. The bit-flip kernels are seeded
from the clock, so results under bit errors vary from run to run even with a fixed `--seed`; use `--test-errors`
to repeat each error rate. `--deterministic` makes training itself reproducible.

## Experiments of the MCEL paper

`scripts/run_paper_fig3.py` runs the accuracy-over-bit-error-rate experiments of Fig. 3 of
[MCEL: Margin-Based Cross-Entropy Loss for Error-Tolerant Quantized Neural Networks](https://arxiv.org/abs/2603.05048)
(2-, 4- and 8-bit QNNs and BNNs on FashionMNIST/VGG3, SVHN/VGG7, CIFAR10/MobileNetV2 and Imagenette/ResNet18, each
with CEL, MCEL and, for BNNs, MHL), with the hyperparameters of Table II of the paper. Finished runs are skipped when
the script is started again. `scripts/plot_error_tolerance.py` plots the results in the style of Fig. 3 and writes
tikz data files and a summary table.

```
python3 scripts/run_paper_fig3.py --print-only                     # list all commands
python3 scripts/run_paper_fig3.py --profile smoke                  # every case once, 1 training batch (~30 min)
python3 scripts/run_paper_fig3.py --profile quick --only qnn4_     # 4-bit networks, 10 epochs (~1 h)
python3 scripts/run_paper_fig3.py                                  # all experiments (~60 h on a GTX 1060)
python3 scripts/plot_error_tolerance.py paper_runs                 # figure, tikz data and summary
```

## Code structure

* `run_master.py`: command line, training and bit-error evaluation
* `code/python/models.py`: all networks (`get_model(name, type, bits)`) and their model-specific defaults
* `code/python/Datasets.py`: datasets and data loaders
* `code/python/TrainTest.py`: training, testing and the bit-error sweep
* `code/python/QuantMethods.py`: wrappers around the binarization/quantization and error-injection kernels
* `code/python/QuantizedNN.py`, `code/python/Utils.py`: quantized layers, losses and helpers
* `code/cuda/`: CUDA kernels
* `scripts/`: experiments of the MCEL paper and plotting

The code is an extension based on the MNIST example in https://github.com/pytorch/examples/tree/master/mnist.

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
