import torch
import torch.nn as nn
import torch.nn.functional as F
import binarizePM1
import quantization

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Quantization1:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)
    
class Quantization2:
    def __init__(self, method, bits=None, unsign=1):
        self.method = method
        self.bits = bits
        self.unsigned = unsign # 0: use signed, 1: use unsigned
    def applyQuantization(self, input):
        return self.method(input,
         input.min().item(), input.max().item(), self.bits, self.unsigned)

binarizepm1 = Quantization1(binarizePM1.binarize)

q_train = True # quantization during training
q_eval = True # quantization during evaluation


class VGG3(nn.Module):
    def __init__(self, cel_train, cel_test, weightBits=None, inputBits=None, quantize_train=True, quantize_eval=True):
        super(VGG3, self).__init__()
        self.relu = nn.ReLU()
        self.name = "VGG3"
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.traincriterion = cel_train
        self.testcriterion = cel_test
        self.weight = weightBits
        self.input = inputBits

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=self.weight, quantize_train=q_train, quantize_eval=q_eval)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 2
        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=self.weight, quantize_train=q_train, quantize_eval=q_eval)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 3
        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=self.weight, quantize_train=q_train, quantize_eval=q_eval)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        self.fc2 = QuantizedLinear(2048, 10, quantization=self.weight, quantize_train=q_train, quantize_eval=q_eval)
        self.scale = Scale()

    def forward(self, x):

        # block 1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.qact1(x)

        # block 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.qact2(x)

        # block 3
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.qact3(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x

class VGG7(nn.Module):
    def __init__(self, cel_train, cel_test, weightBits=None, inputBits=None, quantize_train=True, quantize_eval=True):
        super(VGG7, self).__init__()
        self.relu = nn.ReLU()
        self.name = "VGG7"
        self.weight = weightBits
        self.input = inputBits
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.traincriterion = cel_train
        self.testcriterion = cel_test

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(3, 128, kernel_size=3, padding=1, stride=1, quantization=self.weight, layerNr=1, quantize_train=q_train, quantize_eval=q_eval)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 2
        self.conv2 = QuantizedConv2d(128, 128, kernel_size=3, padding=1, stride=1, quantization=self.weight, layerNr=2, quantize_train=q_train, quantize_eval=q_eval)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 3
        self.conv3 = QuantizedConv2d(128, 256, kernel_size=3, padding=1, stride=1, quantization=self.weight, layerNr=3, quantize_train=q_train, quantize_eval=q_eval)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 4
        self.conv4 = QuantizedConv2d(256, 256, kernel_size=3, padding=1, stride=1, quantization=self.weight, layerNr=4, quantize_train=q_train, quantize_eval=q_eval)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 5
        self.conv5 = QuantizedConv2d(256, 512, kernel_size=3, padding=1, stride=1, quantization=self.weight, layerNr=5, quantize_train=q_train, quantize_eval=q_eval)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 6
        self.conv6 = QuantizedConv2d(512, 512, kernel_size=3, padding=1, stride=1, quantization=self.weight, layerNr=6, quantize_train=q_train, quantize_eval=q_eval)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        # block 7
        self.fc1 = QuantizedLinear(8192, 1024, quantization=self.weight, layerNr=7, quantize_train=q_train, quantize_eval=q_eval)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation(quantization=self.input, quantize_train=q_train, quantize_eval=q_eval)

        self.fc2 = QuantizedLinear(1024, 10, quantization=self.weight, layerNr=8, quantize_train=q_train, quantize_eval=q_eval)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x):

        # block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.qact1(x)

        # block 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.qact2(x)

        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.qact3(x)

        # block 4
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.qact4(x)

        # block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.qact5(x)

        # block 6
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.qact6(x)

        # block 7
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.qact3(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, weightBits=None, inputBits=None, an_sim=None, array_size=None, mapping=None, mapping_distr=None, sorted_mapping_idx=None, performance_mode=None, quantize_train=True, quantize_eval=True, error_model=None, train_model=None, extract_absfreq=None):
        super(BasicBlock, self).__init__()
        self.htanh = nn.Hardtanh()
        self.qact = QuantizedActivation(quantization=inputBits)
        self.conv1 = QuantizedConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, quantization=weightBits, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
            performance_mode=performance_mode,
            error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, quantization=weightBits, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                               performance_mode=performance_mode,
                               error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuantizedConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, quantization=weightBits, an_sim=an_sim, array_size=array_size, mac_mapping=mapping, mac_mapping_distr=mapping_distr, sorted_mac_mapping_idx=sorted_mapping_idx,
                          performance_mode=performance_mode,
                          error_model=error_model, bias=False, train_model=train_model, extract_absfreq=extract_absfreq),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.qact(self.htanh(out))
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, cel_train, cel_test, quantize_train=True, quantize_eval=True, train_model=None, num_classes=10):
        super(ResNet, self).__init__()
        self.name = "ResNet"
        self.traincriterion = cel_train
        self.testcriterion = cel_test
        self.q_train = quantize_train
        self.q_test = quantize_eval
        self.train_model = train_model
        self.in_planes = 64

        self.htanh = nn.Hardtanh()
        self.qact = QuantizedActivation(quantization=self.input)

        self.conv1 = QuantizedConv2d(3, 64, kernel_size=3, stride=1, padding=1, quantization=self.weight)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = QuantizedLinear(512*block.expansion, num_classes, quantization=self.weight, train_model=self.train_model)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, quantMethod=self.quantization, train_model=self.train_model))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.qact(self.htanh(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        # out = F.max_pool2d(out, 2)
        out = self.layer2(out)
        # out = F.max_pool2d(out, 2)
        out = self.layer3(out)
        # out = F.max_pool2d(out, 2)
        out = F.max_pool2d(out, 2)
        out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    