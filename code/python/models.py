"""All network definitions and their model-dependent defaults.

Binarized (BNN) and quantized (QNN) variants of an architecture are separate
classes because their structure differs (BNN: Hardtanh + QuantizedActivation,
QNN: ReLU, weights-only quantization). QNN classes take the bit width as a
constructor argument.

Use `get_model(name, kind, bits, ...)` to build a network; it validates the
combination and raises a ValueError listing the valid options otherwise.
`MODEL_REGISTRY` holds every model-dependent detail (default dataset, default bit
width, optimizer, error-rate grid, recommended hyperparameters, ...).

The quantized layers (QuantizedConv2d / QuantizedLinear / QuantizedActivation)
live in QuantizedNN.py, Scale in Utils.py, and the quantization / error-model
wrappers in QuantMethods.py; they are imported, not duplicated.

The classes replace the model classes that were defined inline in the former
run_*.py scripts and in the former Models.py (both removed). Layer attribute
names and layer creation order are kept identical to those, so that
  * the same --seed produces the same initial weights, and
  * state_dicts saved by the former scripts can be loaded into these classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation
from Utils import Scale
from QuantMethods import make_binary_methods, make_quantized_methods, VALID_BITS

BINARY = "binary"
QUANTIZED = "quantized"
KINDS = (BINARY, QUANTIZED)


# --------------------------------------------------------------------------
# Base classes
# --------------------------------------------------------------------------
class QuantizedNet(nn.Module):
    """Common plumbing: holds quantization + error model, builds quantized layers."""
    kind = None

    def __init__(self, quantization, error_model, quantize_train=True, quantize_eval=True, name=None):
        super(QuantizedNet, self).__init__()
        self.name = name if name is not None else type(self).__name__
        self.quantization = quantization
        self.error_model = error_model
        self.q_train = quantize_train
        self.q_test = quantize_eval

    @property
    def method(self):
        """Same dict the former run scripts printed as 'Error model'."""
        return {"type": "flip", "p": self.error_model.p}

    def qconv(self, *args, **kwargs):
        return QuantizedConv2d(*args, quantization=self.quantization, error_model=self.error_model,
                               quantize_train=self.q_train, quantize_eval=self.q_test, **kwargs)

    def qlinear(self, *args, **kwargs):
        return QuantizedLinear(*args, quantization=self.quantization, error_model=self.error_model,
                               quantize_train=self.q_train, quantize_eval=self.q_test, **kwargs)

    def qact(self):
        # activations are quantized but (as in all former scripts) not error-injected
        return QuantizedActivation(quantization=self.quantization,
                                   quantize_train=self.q_train, quantize_eval=self.q_test)

    def set_bn_momentum(self, momentum):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = momentum


class BinarizedNet(QuantizedNet):
    """Base for {-1,+1} binarized networks (binarizePM1 / binarizePM1FI kernels)."""
    kind = BINARY

    def __init__(self, quantize_train=True, quantize_eval=True, name=None):
        quant, errors = make_binary_methods()
        super(BinarizedNet, self).__init__(quant, errors, quantize_train, quantize_eval, name)
        self.bits = 1


class UniformQuantizedNet(QuantizedNet):
    """Base for uniformly quantized networks with `bits`-bit weights."""
    kind = QUANTIZED

    def __init__(self, bits, quantize_train=True, quantize_eval=True, name=None):
        quant, errors = make_quantized_methods(bits)
        super(UniformQuantizedNet, self).__init__(quant, errors, quantize_train, quantize_eval, name)
        self.bits = bits


# --------------------------------------------------------------------------
# VGG3 (FashionMNIST)
# --------------------------------------------------------------------------
class BNN_VGG3(BinarizedNet):
    """Originally QNN_FMNIST (binarized despite its name) in the removed
    run_vgg3_fashion_quantized.py, and VGG3 in the removed Models.py."""

    def __init__(self, num_classes=10, in_channels=1, input_size=28, scale_init=1e-3,
                 quantize_train=True, quantize_eval=True, name="BNN_FMNIST"):
        super(BNN_VGG3, self).__init__(quantize_train, quantize_eval, name)
        self.htanh = nn.Hardtanh()
        feat = (input_size // 4) ** 2 * 64

        self.conv1 = self.qconv(in_channels, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = self.qact()

        self.conv2 = self.qconv(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = self.qact()

        self.fc1 = self.qlinear(feat, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = self.qact()

        self.fc2 = self.qlinear(2048, num_classes)
        self.scale = Scale(init_value=scale_init)

    def forward(self, x):
        x = self.qact1(self.htanh(self.bn1(F.max_pool2d(self.conv1(x), 2))))
        x = self.qact2(self.htanh(self.bn2(F.max_pool2d(self.conv2(x), 2))))
        x = torch.flatten(x, 1)
        x = self.qact3(self.htanh(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return self.scale(x)


class QNN_VGG3(UniformQuantizedNet):
    """New: built from a commented-out 4-bit QNN_FMNIST in the removed
    run_vgg3_fashion_quantized.py; never trained by the former scripts."""

    def __init__(self, bits=4, num_classes=10, in_channels=1, input_size=28, scale_init=1e-3,
                 quantize_train=True, quantize_eval=True, name="QNN_FMNIST"):
        super(QNN_VGG3, self).__init__(bits, quantize_train, quantize_eval, name)
        self.relu = nn.ReLU()
        feat = (input_size // 4) ** 2 * 64

        self.conv1 = self.qconv(in_channels, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = self.qconv(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = self.qlinear(feat, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc2 = self.qlinear(2048, num_classes)
        self.scale = Scale(init_value=scale_init)

    def forward(self, x):
        x = self.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = self.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = torch.flatten(x, 1)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return self.scale(x)


# --------------------------------------------------------------------------
# VGG7 (SVHN)
# --------------------------------------------------------------------------
VGG7_CHANNELS = (64, 64, 128, 128, 256, 256)
VGG7_POOL_AFTER = (2, 4, 6)


class BNN_VGG7(BinarizedNet):
    """Originally BNN_VGG7_SVHN in the removed run_vgg7_svhn_binarized.py."""

    def __init__(self, num_classes=10, in_channels=3, input_size=32, scale_init=1e-5,
                 quantize_train=True, quantize_eval=True, name="BNN_VGG7_SVHN"):
        super(BNN_VGG7, self).__init__(quantize_train, quantize_eval, name)
        self.htanh = nn.Hardtanh()
        cin = in_channels
        for i, cout in enumerate(VGG7_CHANNELS, 1):
            setattr(self, "conv%d" % i, self.qconv(cin, cout, kernel_size=3, padding=1, stride=1))
            setattr(self, "bn%d" % i, nn.BatchNorm2d(cout))
            setattr(self, "qact%d" % i, self.qact())
            cin = cout
        self.fc1 = self.qlinear(cin * (input_size // 8) ** 2, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.qact7 = self.qact()
        self.fc2 = self.qlinear(512, num_classes)
        self.scale = Scale(init_value=scale_init)

    def forward(self, x):
        for i in range(1, len(VGG7_CHANNELS) + 1):
            x = getattr(self, "conv%d" % i)(x)
            x = getattr(self, "bn%d" % i)(x)
            x = self.htanh(x)
            x = getattr(self, "qact%d" % i)(x)
            if i in VGG7_POOL_AFTER:
                x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.qact7(self.htanh(self.bn7(self.fc1(x))))
        x = self.fc2(x)
        return self.scale(x)


class QNN_VGG7(UniformQuantizedNet):
    """Originally QNN_VGG7_SVHN in the removed run_vgg7_svhn_quantized.py."""

    def __init__(self, bits=8, num_classes=10, in_channels=3, input_size=32, scale_init=1e-3,
                 quantize_train=True, quantize_eval=True, name="QNN_VGG7_SVHN"):
        super(QNN_VGG7, self).__init__(bits, quantize_train, quantize_eval, name)
        self.relu = nn.ReLU(inplace=True)
        cin = in_channels
        for i, cout in enumerate(VGG7_CHANNELS, 1):
            setattr(self, "conv%d" % i, self.qconv(cin, cout, kernel_size=3, padding=1, stride=1))
            setattr(self, "bn%d" % i, nn.BatchNorm2d(cout))
            cin = cout
        self.fc1 = self.qlinear(cin * (input_size // 8) ** 2, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.fc2 = self.qlinear(512, num_classes)
        self.scale = Scale(init_value=scale_init)

    def forward(self, x):
        for i in range(1, len(VGG7_CHANNELS) + 1):
            x = getattr(self, "conv%d" % i)(x)
            x = getattr(self, "bn%d" % i)(x)
            x = self.relu(x)
            if i in VGG7_POOL_AFTER:
                x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.relu(self.bn7(self.fc1(x)))
        x = self.fc2(x)
        return self.scale(x)


# --------------------------------------------------------------------------
# Wide VGG7 (CIFAR10) and fully connected BNN
# --------------------------------------------------------------------------
VGG7_WIDE_CHANNELS = (128, 128, 256, 256, 512, 512)


class BNN_VGG7Wide(BinarizedNet):
    """Originally VGG7 in the removed Models.py (used by the removed run.py).

    Unlike there, the error model is attached to the conv/linear layers so that
    the bit-error sweep works for this model too.
    """

    def __init__(self, num_classes=10, in_channels=3, input_size=32, scale_init=1e-3,
                 quantize_train=True, quantize_eval=True, name="VGG7"):
        super(BNN_VGG7Wide, self).__init__(quantize_train, quantize_eval, name)
        self.htanh = nn.Hardtanh()
        cin = in_channels
        for i, cout in enumerate(VGG7_WIDE_CHANNELS, 1):
            setattr(self, "conv%d" % i, self.qconv(cin, cout, kernel_size=3, padding=1, stride=1, layerNr=i))
            setattr(self, "bn%d" % i, nn.BatchNorm2d(cout))
            setattr(self, "qact%d" % i, self.qact())
            cin = cout
        self.fc1 = self.qlinear(cin * (input_size // 8) ** 2, 1024, layerNr=7)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = self.qact()
        self.fc2 = self.qlinear(1024, num_classes, layerNr=8)
        self.scale = Scale(init_value=scale_init)

    def forward(self, x):
        # unlike BNN_VGG7, this model pools directly after the conv (before BN) in blocks 2, 4, 6
        for i in range(1, len(VGG7_WIDE_CHANNELS) + 1):
            x = getattr(self, "conv%d" % i)(x)
            if i in VGG7_POOL_AFTER:
                x = F.max_pool2d(x, 2)
            x = getattr(self, "bn%d" % i)(x)
            x = self.htanh(x)
            x = getattr(self, "qact%d" % i)(x)
        x = torch.flatten(x, 1)
        x = self.qact7(self.htanh(self.bn7(self.fc1(x))))
        x = self.fc2(x)
        return self.scale(x)


class BNN_FC(BinarizedNet):
    """3-layer fully connected BNN. Originally BNN_FASHION_FC in the removed
    Models.py (not used by any former script). The error model is attached to
    the linear layers (unlike there).
    """

    def __init__(self, num_classes=10, in_channels=1, input_size=28, scale_init=1e-3,
                 quantize_train=True, quantize_eval=True, name="BNN_FASHION_FC"):
        super(BNN_FC, self).__init__(quantize_train, quantize_eval, name)
        self.htanh = nn.Hardtanh()
        self.fcfc1 = self.qlinear(in_channels * input_size * input_size, 2048, layerNr=1)
        self.fcbn1 = nn.BatchNorm1d(2048)
        self.fcqact1 = self.qact()
        self.fcfc2 = self.qlinear(2048, 2048, layerNr=2)
        self.fcbn2 = nn.BatchNorm1d(2048)
        self.fcqact2 = self.qact()
        self.fcfc3 = self.qlinear(2048, num_classes, layerNr=3)
        self.scale = Scale(init_value=scale_init)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fcqact1(self.htanh(self.fcbn1(self.fcfc1(x))))
        x = self.fcqact2(self.htanh(self.fcbn2(self.fcfc2(x))))
        x = self.fcfc3(x)
        return self.scale(x)


# --------------------------------------------------------------------------
# ResNet-18 (ImageNette 64x64)
# --------------------------------------------------------------------------
RESNET18_CHANNELS = (64, 128, 256, 512)


class QNN_ResNet18(UniformQuantizedNet):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool, 1x1 strided downsample convs).

    Originally QNN_ResNet18_Imagenette_64 in the removed
    run_resnet18_imagenette_quantized.py.
    `width` multiplies every channel count (default 1, as in the original).
    Layer names: stage s in {2,3,4,5} has convs conv<s>_1..conv<s>_4 / bn<s>_1..4;
    stages 3-5 have a projection shortcut down<s> / down<s>_bn.
    """

    def __init__(self, bits=4, num_classes=10, in_channels=3, input_size=64, width=1, scale_init=1e-3,
                 quantize_train=True, quantize_eval=True, name="QNN_ResNet18_Imagenette_64"):
        super(QNN_ResNet18, self).__init__(bits, quantize_train, quantize_eval, name)
        self.relu = nn.ReLU(inplace=True)
        ch = [width * c for c in RESNET18_CHANNELS]

        # stem
        self.conv1 = self.qconv(in_channels, ch[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch[0])
        # stage 2 (identity shortcuts); creation order as in the former script
        for j in range(1, 5):
            setattr(self, "conv2_%d" % j, self.qconv(ch[0], ch[0], 3, 1, 1))
            setattr(self, "bn2_%d" % j, nn.BatchNorm2d(ch[0]))
        # stages 3..5 (first block downsamples)
        for s, (cin, cout) in zip((3, 4, 5), zip(ch[:-1], ch[1:])):
            setattr(self, "conv%d_1" % s, self.qconv(cin, cout, 3, 2, 1))
            setattr(self, "bn%d_1" % s, nn.BatchNorm2d(cout))
            setattr(self, "conv%d_2" % s, self.qconv(cout, cout, 3, 1, 1))
            setattr(self, "bn%d_2" % s, nn.BatchNorm2d(cout))
            setattr(self, "down%d" % s, self.qconv(cin, cout, 1, 2, 0))
            setattr(self, "down%d_bn" % s, nn.BatchNorm2d(cout))
            setattr(self, "conv%d_3" % s, self.qconv(cout, cout, 3, 1, 1))
            setattr(self, "bn%d_3" % s, nn.BatchNorm2d(cout))
            setattr(self, "conv%d_4" % s, self.qconv(cout, cout, 3, 1, 1))
            setattr(self, "bn%d_4" % s, nn.BatchNorm2d(cout))
        # head
        self.fc = self.qlinear(ch[-1], num_classes)
        self.scale = Scale(init_value=scale_init)

    def _block(self, x, s, first, shortcut):
        a, b = (1, 2) if first else (3, 4)
        r = x if shortcut is None else shortcut(x)
        x = self.relu(getattr(self, "bn%d_%d" % (s, a))(getattr(self, "conv%d_%d" % (s, a))(x)))
        x = getattr(self, "bn%d_%d" % (s, b))(getattr(self, "conv%d_%d" % (s, b))(x))
        return self.relu(x + r)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for s in (2, 3, 4, 5):
            shortcut = None
            if s > 2:
                down, down_bn = getattr(self, "down%d" % s), getattr(self, "down%d_bn" % s)
                shortcut = lambda t, d=down, bn=down_bn: bn(d(t))  # noqa: E731
            x = self._block(x, s, True, shortcut)
            x = self._block(x, s, False, None)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.scale(x)


# --------------------------------------------------------------------------
# Reduced MobileNetV2 (CIFAR10)
# --------------------------------------------------------------------------
class QNN_MobileNetV2(UniformQuantizedNet):
    """Stem + 3 inverted-residual blocks + 1x1 head conv + FC.

    Originally QNN_MobileNetV2_CIFAR10 in the removed
    run_mobilenetv2_cifar10_quantized.py (head_width=1) and
    run_mobilenetv2_cifar10_quantized_more_neurons.py (head_width=4).
    `width` multiplies the body channels, `head_width` the 256-channel head.
    BatchNorm momentum is set to `bn_momentum` (0.01, as in the former scripts).
    """

    def __init__(self, bits=4, num_classes=10, in_channels=3, input_size=32, width=1, head_width=1,
                 bn_momentum=0.01, scale_init=1e-3, quantize_train=True, quantize_eval=True,
                 name="QNN_MobileNetV2_CIFAR10"):
        super(QNN_MobileNetV2, self).__init__(bits, quantize_train, quantize_eval, name)
        self.relu = nn.ReLU(inplace=True)
        c32, c64, c96, c192, c384 = [width * c for c in (32, 64, 96, 192, 384)]
        chead = head_width * 256

        # stem
        self.conv1 = self.qconv(in_channels, c32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c32)
        # block 1 (t=1, c=32, residual)
        self.dw2 = self.qconv(c32, c32, kernel_size=3, stride=1, padding=1, groups=c32)
        self.bn2 = nn.BatchNorm2d(c32)
        self.pw2 = self.qconv(c32, c32, kernel_size=1, stride=1, padding=0)
        self.bn2p = nn.BatchNorm2d(c32)
        # block 2 (t=6, c=64, stride 2)
        self.pw3_1 = self.qconv(c32, c192, 1, 1, 0)
        self.bn3_1 = nn.BatchNorm2d(c192)
        self.dw3 = self.qconv(c192, c192, 3, 2, 1, groups=c192)
        self.bn3_2 = nn.BatchNorm2d(c192)
        self.pw3_2 = self.qconv(c192, c64, 1, 1, 0)
        self.bn3_3 = nn.BatchNorm2d(c64)
        # block 3 (t=6, c=96)
        self.pw4_1 = self.qconv(c64, c384, 1, 1, 0)
        self.bn4_1 = nn.BatchNorm2d(c384)
        self.dw4 = self.qconv(c384, c384, 3, 1, 1, groups=c384)
        self.bn4_2 = nn.BatchNorm2d(c384)
        self.pw4_2 = self.qconv(c384, c96, 1, 1, 0)
        self.bn4_3 = nn.BatchNorm2d(c96)
        # head
        self.conv5 = self.qconv(c96, chead, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(chead)
        self.fc = self.qlinear(chead, num_classes)
        self.scale = Scale(init_value=scale_init)

        if bn_momentum is not None:
            self.set_bn_momentum(bn_momentum)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # block 1 (residual)
        y = self.relu(self.bn2(self.dw2(x)))
        y = self.bn2p(self.pw2(y))
        x = x + y
        # block 2
        x = self.relu(self.bn3_1(self.pw3_1(x)))
        x = self.relu(self.bn3_2(self.dw3(x)))
        x = self.bn3_3(self.pw3_2(x))
        # block 3
        x = self.relu(self.bn4_1(self.pw4_1(x)))
        x = self.relu(self.bn4_2(self.dw4(x)))
        x = self.bn4_3(self.pw4_2(x))
        # head
        x = self.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.scale(x)


# --------------------------------------------------------------------------
# Registry of model-dependent details
# --------------------------------------------------------------------------
def error_grid(divisor, count):
    """Bit-error rates [j/divisor for j in range(count)], as in the former scripts."""
    return [j / divisor for j in range(count)]


class ModelSpec(object):
    """Everything that depends on (architecture, kind)."""

    def __init__(self, cls, default_dataset, model_kwargs=None, default_bits=None, optimizer="adam",
                 hparams=None, error_rates=None):
        self.cls = cls
        self.default_dataset = default_dataset
        self.model_kwargs = dict(model_kwargs or {})
        self.default_bits = default_bits
        self.optimizer = optimizer          # "adam" or "clippy"
        self.hparams = dict(hparams or {})  # recommended batch_size/epochs/lr/step_size/gamma
        self.error_rates = error_rates      # default bit-error sweep grid


def _hp(batch_size, epochs, lr, step_size, gamma):
    return {"batch_size": batch_size, "epochs": epochs, "lr": lr, "step_size": step_size, "gamma": gamma}


MODEL_REGISTRY = {
    "vgg3": {
        BINARY: ModelSpec(BNN_VGG3, "FMNIST", optimizer="clippy",
                          hparams=_hp(256, 50, 0.001, 5, 0.5), error_rates=error_grid(100, 31)),
        QUANTIZED: ModelSpec(QNN_VGG3, "FMNIST", default_bits=4, optimizer="clippy",
                             hparams=_hp(256, 10, 0.0001, 5, 0.5), error_rates=error_grid(100, 31)),
    },
    "vgg7": {
        BINARY: ModelSpec(BNN_VGG7, "SVHN", hparams=_hp(256, 100, 0.001, 10, 0.5),
                          error_rates=error_grid(200, 41)),
        QUANTIZED: ModelSpec(QNN_VGG7, "SVHN", default_bits=8, hparams=_hp(256, 50, 0.001, 5, 0.25),
                             error_rates=error_grid(2000, 41)),
    },
    "vgg7_wide": {
        BINARY: ModelSpec(BNN_VGG7Wide, "CIFAR10", optimizer="clippy", hparams=_hp(256, 5, 0.001, 2, 0.5),
                          error_rates=error_grid(200, 41)),
    },
    "fc": {
        BINARY: ModelSpec(BNN_FC, "FMNIST", optimizer="clippy", hparams=_hp(256, 5, 0.001, 2, 0.5),
                          error_rates=error_grid(100, 31)),
    },
    "resnet18": {
        QUANTIZED: ModelSpec(QNN_ResNet18, "IMAGENETTE", default_bits=4, hparams=_hp(64, 50, 0.001, 10, 0.5),
                             error_rates=error_grid(2000, 41)),
    },
    "mobilenetv2": {
        QUANTIZED: ModelSpec(QNN_MobileNetV2, "CIFAR10", default_bits=4, hparams=_hp(128, 100, 0.001, 10, 0.5),
                             error_rates=error_grid(16000, 61)),
    },
    "mobilenetv2_2x": {
        QUANTIZED: ModelSpec(QNN_MobileNetV2, "CIFAR10", default_bits=4,
                             model_kwargs={"head_width": 4, "name": "QNN_MobileNetV2_CIFAR10_2x"},
                             hparams=_hp(128, 100, 0.001, 10, 0.5), error_rates=error_grid(8000, 61)),
    },
}

MODEL_NAMES = tuple(sorted(MODEL_REGISTRY))


def valid_combinations():
    """List of (model, kind) pairs that exist."""
    return [(m, k) for m in MODEL_NAMES for k in KINDS if k in MODEL_REGISTRY[m]]


def _combo_text():
    return ", ".join("{}/{}".format(m, k) for m, k in valid_combinations())


def get_spec(name, kind):
    """Return the ModelSpec for (name, kind) or raise ValueError listing valid options.

    name and kind are case-insensitive.
    """
    key = str(name).lower()
    if key not in MODEL_REGISTRY:
        raise ValueError("Unknown model {!r}. Valid models: {}".format(name, ", ".join(MODEL_NAMES)))
    kind = str(kind).lower()
    if kind not in KINDS:
        raise ValueError("Unknown type {!r}. Valid types: {}".format(kind, ", ".join(KINDS)))
    if kind not in MODEL_REGISTRY[key]:
        raise ValueError("Model {!r} has no {} variant. Valid model/type combinations: {}".format(
            name, kind, _combo_text()))
    return MODEL_REGISTRY[key][kind]


def resolve_bits(name, kind, bits):
    """Validate the bit width for (name, kind); returns the effective bit width (None for binary)."""
    spec = get_spec(name, kind)
    if str(kind).lower() == BINARY:
        if bits is not None:
            raise ValueError("--bits/bits is only valid for quantized models; {} is binary "
                             "(1 bit, {{-1,+1}}). Remove the bit width or use type 'quantized'.".format(name))
        return None
    if bits is None:
        bits = spec.default_bits
    if bits not in VALID_BITS:
        raise ValueError("Unsupported bit width {!r} for quantized {}. Valid bit widths: {}".format(
            bits, name, ", ".join(str(b) for b in VALID_BITS)))
    return bits


def get_model(name, kind, bits=None, num_classes=10, in_channels=None, input_size=None,
              quantize_train=True, quantize_eval=True, **overrides):
    """Build a network.

    name:  architecture (see MODEL_NAMES)
    kind:  'binary' or 'quantized'
    bits:  bit width for quantized models (default: the model's default_bits);
           must be None for binary models.
    num_classes / in_channels / input_size: dataset geometry; in_channels and
           input_size default to the class defaults (the model's default dataset).
    overrides: extra constructor kwargs (e.g. scale_init, width, head_width, name).
    """
    kind = str(kind).lower()
    spec = get_spec(name, kind)
    bits = resolve_bits(name, kind, bits)
    kwargs = dict(spec.model_kwargs)
    kwargs.update(num_classes=num_classes, quantize_train=quantize_train, quantize_eval=quantize_eval)
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if input_size is not None:
        kwargs["input_size"] = input_size
    kwargs.update(overrides)
    if kind == QUANTIZED:
        kwargs["bits"] = bits
    return spec.cls(**kwargs)


def default_save_name(name, kind, bits=None):
    """Default checkpoint file name, unique per (model, type, bits), e.g. 'vgg7_quantized_8bit.pt'."""
    if bits is None:
        return "{}_{}.pt".format(str(name).lower(), str(kind).lower())
    return "{}_{}_{}bit.pt".format(str(name).lower(), str(kind).lower(), bits)
