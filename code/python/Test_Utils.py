import torch
import torch.nn as nn
import argparse
import os
from datetime import datetime
import json

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation
from Test_Models import VGG3_Test_Q, VGG3_Test_B #, VGG3_BNN, VGG3_QI2, VGG3_QI4, VGG3_QI8, VGG7_BNN, VGG7_QI2, VGG7_QI4, VGG7_QI8, ResNet_BNN, ResNet_QI2, ResNet_QI4, ResNet_QI8

def set_layer_mode(model, mode):
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if mode == "train":
                layer.training = True
            if mode == "eval":
                layer.eval = False

def parse_args(parser):
    parser.add_argument('--dataset', type=str, default=None,
                    help='MNIST/FMNIST/QMNIST/SVHN/CIFAR10')
    parser.add_argument('--train-model', type=int, default=None, help='Whether to train a model')
    parser.add_argument('--load-model-path', type=str, default=None, help='Specify path to model if it should be loaded')
    parser.add_argument('--gpu-num', type=int, default=0, metavar='N', help='Specify the GPU on which the training should be performed')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--step-size', type=int, default=25, metavar='M',
                        help='Learning step size (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--test-error', action='store_true', default=False,
                        help='Test accuracy under errors')
    parser.add_argument('--weight', type=int, default=4, help='Quanization of weight-bits (default: 4)')
    parser.add_argument('--input', type=int, default=4, help='Quanization of input-bits (default: 4)')


def dump_exp_data(model, args, all_accuracies):
    to_dump = dict()
    to_dump["model"] = model.name
    # to_dump["method"] = model.method
    to_dump["batchsize"] = args.batch_size
    to_dump["epochs"] = args.epochs
    to_dump["learning_rate"] = args.lr
    to_dump["gamma"] = args.gamma
    to_dump["stepsize"] = args.step_size
    # to_dump["traincrit"] = model.traincriterion.name
    # to_dump["testcrit"] = model.testcriterion.name
    to_dump["test_error"] = all_accuracies
    to_dump["weight"] = args.weight
    to_dump["input"] = args.input
    return to_dump

def create_exp_folder(model):
    exp_path = ""
    access_rights = 0o755
    this_path = os.getcwd()
    exp_path += this_path+"/experiments/"+model.name+"/"+"results-"+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
    try:
        os.makedirs(exp_path, access_rights, exist_ok=False)
    except OSError:
        print ("Creation of the directory %s failed" % exp_path)
    else:
        print ("Successfully created the directory %s" % exp_path)
    return exp_path + "/results.jsonl"

def store_exp_data(to_dump_path, to_dump_data):
    with open(to_dump_path, 'a') as outfile:
        json.dump(to_dump_data, outfile)
        print ("Successfully stored results in %s" % to_dump_path)

# TODO: ONNX save
