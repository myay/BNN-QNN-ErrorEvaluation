import torch
import quantization
import binarizePM1

class Quantization1:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)

binarizepm1 = Quantization1(binarizePM1.binarize)

tensor = torch.rand(size=(2,2,3,3), dtype=torch.float).cuda()
tensor *= 100
tensor -= torch.mean(tensor)
print("random tensor", tensor)
a = quantization.quantize(tensor, tensor.min().item(), tensor.max().item(), 8, 0)
print("signed quantization", a)
b = quantization.quantize(tensor, tensor.min().item(), tensor.max().item(), 8, 1)
print("unsigned quantization", b)
c = quantization.quantize(tensor, tensor.min().item(), tensor.max().item(), 4, 1)
print("4 bit quantization", c)
print("------------------------------------")
print("'requantize' to 8 bit: ", b)
#'requantization' with this approach without success!
x = quantization.quantize(c, c.min().item(), c.max().item(), 8, 1)
print("another try of 'requantization' of c ",  x)
#successfully applied quantization in 'another try'!

print("------------------------------------")
print("Requantization to 4 bits:")

print("simple approach: ", c)
c = quantization.quantize(x, x.min().item(), x.max().item(), 4, 1)
print("alternative approach: ", c)

#conclusion: quantization has always to be made 'manually'.

print("------------------------------------")
print("------------------------------------")
print("------------------------------------")
print("Binarization")

bb = binarizePM1.binarize(tensor)
print ("binarization: ", bb)


#print("c.bits: ", c.dtype)
#print("c.unsigned: ", c.sign)

# b = qtize.shift_bfi_4bit(a, 0.1, 0.1)
# print(b)
