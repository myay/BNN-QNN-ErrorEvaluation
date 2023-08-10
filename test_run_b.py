from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import json
import sys
import os
from datetime import datetime
sys.path.append("code/python/")

from Test_Utils import set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Test_Models import VGG3_Test_B

from Traintest_Utils import train, test, test_error, Criterion, binary_hingeloss, Clippy

import binarizePM1
import binarizePM1FI
import quantization
import quantizationFI

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
        output = self.method_enc_dec(input,
         input.min().item(), input.max().item(), self.bits, 1) # to unsigned, encode
        output = self.method_errors(input, self.p, self.p, self.bits) # inject bit flips
        output = self.method_enc_dec(input,
         input.min().item(), input.max().item(), self.bits, 0) # to signed again, decode
        return output
    
class Quantization1:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)
    
binarizepm1 = Quantization1(binarizePM1.binarize)

class Quantization2:
    def __init__(self, method, bits=None, unsign=0):
        self.method = method
        self.bits = bits
        self.unsigned = unsign # 0: use signed, 1: use unsigned
    def applyQuantization(self, input):
        return self.method(input,
         input.min().item(), input.max().item(), self.bits, self.unsigned)
    
# q4bit = Quantization2(quantization.quantize, 4, 0)

cel_train = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_train")
cel_test = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_test")

q_train = True # quantization during training
q_eval = True # quantization during evaluation

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training Process')
    parse_args(parser)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.FashionMNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.FashionMNIST('data', train=False,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    nn_model = VGG3_Test_B

    tensor = torch.rand(size=(2,2,3,3), dtype=torch.float).cuda()

    model = nn_model(cel_train, cel_test, weightBits=binarizepm1, inputBits=binarizepm1, quantize_train=q_train, quantize_eval=q_eval).to(device)

    # create experiment folder and file
    to_dump_path = create_exp_folder(model)
    if not os.path.exists(to_dump_path):
        open(to_dump_path, 'w').close()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Clippy(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    time_elapsed = 0
    times = []
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        since = int(round(time.time()*1000))
        #
        train(args, model, device, train_loader, optimizer, epoch)
        #
        time_elapsed += int(round(time.time()*1000)) - since
        print('Epoch training time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
        # test(model, device, train_loader)
        since = int(round(time.time()*1000))
        #
        test(model, device, test_loader)
        #
        time_elapsed += int(round(time.time()*1000)) - since
        print('Test time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
        # test(model, device, train_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # TODO: ONNX save

if __name__ == '__main__':
    main()
