"""Quantization and bit-error-model wrappers around the CUDA kernels.

These classes were previously copy-pasted at module level into every run_*.py
script. They are moved here unchanged (behaviour is bit-for-bit the same) so
that models.py and run_master.py share a single copy.

The CUDA extensions (binarizePM1, binarizePM1FI, quantization, quantizationFI)
are built from code/cuda/ and must be importable; torch has to be imported
before them.
"""
import torch  # noqa: F401  (must be imported before the CUDA extensions)

import binarizePM1
import binarizePM1FI
import quantization
import quantizationFI


# n-bit QNN error model
class SymmetricBitErrorsQNN:
    def __init__(self, method_errors, method_enc_dec, p, bits, type):
        self.method_errors = method_errors
        self.method_enc_dec = method_enc_dec
        self.p = p
        self.bits = bits
        self.type = type

    def updateErrorModel(self, p_updated):
        self.p = p_updated

    def resetErrorModel(self):
        self.p = 0

    def applyErrorModel(self, input):
        # NOTE: kept exactly as in the original run scripts.
        output = self.method_enc_dec(input,
         input.min().item(), input.max().item(), self.bits, 1) # to unsigned, encode
        output = self.method_errors(input, self.p, self.p, self.bits) # inject bit flips
        output = self.method_enc_dec(input,
         input.min().item(), input.max().item(), self.bits, 0) # to signed again, decode
        return output


class Quantization_uniform:
    def __init__(self, method, bits=None, unsign=0):
        self.method = method
        self.bits = bits
        self.unsigned = unsign # 0: use signed, 1: use unsigned

    def applyQuantization(self, input):
        return self.method(input,
         input.min().item(), input.max().item(), self.bits, self.unsigned)


# BNN error model
class SymmetricBitErrorsBinarizedPM1:
    def __init__(self, method, p):
        self.p = p
        self.method = method

    def updateErrorModel(self, p_updated):
        self.p = p_updated

    def resetErrorModel(self):
        self.p = 0

    def applyErrorModel(self, input):
        return self.method(input, self.p, self.p)


class Quantization1:
    def __init__(self, method):
        self.method = method

    def applyQuantization(self, input):
        return self.method(input)


# The bit-flip kernel works on uint8, so 8 bits is the maximum. 1 bit is the
# binary case and is handled by the BNN classes / binarizePM1 kernels.
MIN_BITS = 2
MAX_BITS = 8
VALID_BITS = tuple(range(MIN_BITS, MAX_BITS + 1))


def make_binary_methods(p=0):
    """Return (quantization, error_model) for {-1,+1} binarized networks.

    Same objects as `binarizepm1` / `binarizepm1fi` in the original scripts.
    """
    quant = Quantization1(binarizePM1.binarize)
    errors = SymmetricBitErrorsBinarizedPM1(binarizePM1FI.binarizeFI, p)
    return quant, errors


def make_quantized_methods(bits, p=0):
    """Return (quantization, error_model) for uniform `bits`-bit quantization.

    Same objects as `q<bits>bit` / `q<bits>bit_errors` in the original scripts
    (unsigned quantization, bit flips injected with quantizationFI.bfi_8bit).
    """
    if bits not in VALID_BITS:
        raise ValueError("Unsupported bit width {!r}. Valid bit widths: {}".format(
            bits, ", ".join(str(b) for b in VALID_BITS)))
    quant = Quantization_uniform(quantization.quantize, bits, 1)
    errors = SymmetricBitErrorsQNN(quantizationFI.bfi_8bit, quantization.quantize, p, bits,
                                   "flip{}bit".format(bits))
    return quant, errors
