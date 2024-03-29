import torch
import torch.nn as nn
import torch.nn.functional as F

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Utils import Scale, Clippy

class VGG3(nn.Module):
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None):
        super(VGG3, self).__init__()
        self.name = "VGG3"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model

        self.htanh = nn.Hardtanh()
        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=self.quantization, error_model=self.error_model)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=self.quantization, error_model=self.error_model)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(2048, 10, quantization=self.quantization, error_model=self.error_model)
        self.scale = Scale()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact2(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x


class VGG7(nn.Module):
    def __init__(self, quantMethod=None, quantize_train=True, quantize_eval=True, error_model=None):
        super(VGG7, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "VGG7"
        self.quantization = quantMethod
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.error_model = error_model

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(3, 128, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation(quantization=self.quantization)

        # block 2
        self.conv2 = QuantizedConv2d(128, 128, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation(quantization=self.quantization)

        # block 3
        self.conv3 = QuantizedConv2d(128, 256, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation(quantization=self.quantization)

        # block 4
        self.conv4 = QuantizedConv2d(256, 256, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=4)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation(quantization=self.quantization)

        # block 5
        self.conv5 = QuantizedConv2d(256, 512, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=5)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation(quantization=self.quantization)

        # block 6
        self.conv6 = QuantizedConv2d(512, 512, kernel_size=3, padding=1, stride=1, quantization=self.quantization, layerNr=6)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation(quantization=self.quantization)

        # block 7
        self.fc1 = QuantizedLinear(8192, 1024, quantization=self.quantization, layerNr=7)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation(quantization=self.quantization)

        self.fc2 = QuantizedLinear(1024, 10, quantization=self.quantization, layerNr=8)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x):

        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        # block 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)

        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact3(x)

        # block 4
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.bn4(x)
        x = self.htanh(x)
        x = self.qact4(x)

        # block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.htanh(x)
        x = self.qact5(x)

        # block 6
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = self.bn6(x)
        x = self.htanh(x)
        x = self.qact6(x)

        # block 7
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)
        x = self.htanh(x)
        x = self.qact3(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x

class BNN_FASHION_FC(nn.Module):
    def __init__(self, quantMethod):
        super(BNN_FASHION_FC, self).__init__()
        self.quantization = quantMethod

        self.htanh = nn.Hardtanh()
        self.flatten = torch.flatten
        self.fcfc1 = QuantizedLinear(28*28, 2048, quantization=self.quantization, layerNr=1)
        self.fcbn1 = nn.BatchNorm1d(2048)
        self.fcqact1 = QuantizedActivation(quantization=self.quantization)

        self.fcfc2 = QuantizedLinear(2048, 2048, quantization=self.quantization, layerNr=2)
        self.fcbn2 = nn.BatchNorm1d(2048)
        self.fcqact2 = QuantizedActivation(quantization=self.quantization)
        self.fcfc3 = QuantizedLinear(2048, 10, quantization=self.quantization, layerNr=3)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x):
        x = self.flatten(x, start_dim=1, end_dim=3)
        x = self.fcfc1(x)
        x = self.fcbn1(x)
        x = self.htanh(x)
        x = self.fcqact1(x)

        x = self.fcfc2(x)
        x = self.fcbn2(x)
        x = self.htanh(x)
        x = self.fcqact2(x)

        x = self.fcfc3(x)
        x = self.scale(x)

        return x
