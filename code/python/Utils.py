import torch
import torch.nn as nn
import argparse
import os
from datetime import datetime
import json

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

def binary_hingeloss(yhat, y, b=128, epoch=0):
   #print("yhat", yhat.mean(dim=1))
   #print("y", y)
   #print("BINHINGE")
   y_enc = 2 * torch.nn.functional.one_hot(y, yhat.shape[-1]) - 1.0
   #print("y_enc", y_enc)
   l = (b - y_enc * yhat).clamp(min=0)
   #print(l)
   return l.mean(dim=1) / b
   
# def margin_softmax_ce(yhat, y, m, epoch):
    # # create one-hot tensor of ground truth
    # #print("am usign msc")
    # ###
    # # some analysis on the NN outputs
    # # print always the largest logits
    # #values = yhat.max(dim=1).values
    # #print("Largest logits", values)
    # #print(yhat[0])
    # ###
    # #print("Epoch in mcel: ", epoch)
    # #if epoch > 0:
    # #print("Now applying margin")
    # #yhat = torch.clamp(yhat, min=-100, max=100)
    
    # #one_hot_labels = torch.nn.functional.one_hot(y, yhat.shape[-1])
    # #margin_mask = -m * one_hot_labels
    # #print("before", yhat[0])
    # #yhat = yhat + margin_mask
    
    # yhat = scale_aware_clamp(yhat, y, m)
    # # print("after", yhat[0])
    # # print("after", one_hot_labels[0])
    # #targets = one_hot_labels.argmax(dim=1)
    
    # #scaling needed for 4 bit QNNs with ReLU 
    # #yhat = yhat*0.01
    # #yhat = yhat*0.1
    # #yhat = torch.zeros(yhat.shape).cuda()
    # #yhat.zero_()
    # #print(yhat[0])
    # return torch.nn.functional.cross_entropy(yhat, y)   
   
def scale_aware_clamp(logits, y, m, eps=1e-6):
    """
    Scale-aware clamping of logits for margin-based losses or BNN/FI networks.

    Args:
        logits: Tensor of shape (B, K) - raw NN outputs
        eps: small constant to avoid division by zero

    Returns:
        Tensor of same shape as logits, clamped in a scale-aware manner
    """
    # 1. compute per-sample standard deviation
    #print(logits[0])
    sigma = logits.std(dim=1, keepdim=True)  # shape (B,1)
    #print(sigma[0])
    sigma = sigma.clamp_min(eps)            # avoid division by zero

    # 2. scale-aware tanh clamping
    logits_clamped = sigma * torch.tanh(logits / sigma)
    #print(logits_clamped[0])
    
    # apply margins
    one_hot_labels = torch.nn.functional.one_hot(y, logits.shape[-1])
    margin_mask = -m * one_hot_labels
    logits_clamped_m_applied = logits_clamped + margin_mask * sigma
    return logits_clamped
   
# def margin_softmax_ce(yhat, y, m, epoch):
    # # create one-hot tensor of ground truth
    # #print("am usign msc")
    # ###
    # # some analysis on the NN outputs
    # # print always the largest logits
    # #values = yhat.max(dim=1).values
    # #print("Largest logits", values)
    # #print(yhat[0])
    # ###
    # #print("Epoch in mcel: ", epoch)
    # #if epoch > 0:
    # #print("Now applying margin")
    # yhat = torch.clamp(yhat, min=-100, max=100)
    # one_hot_labels = torch.nn.functional.one_hot(y, yhat.shape[-1])
    # margin_mask = -m * one_hot_labels
    # #print("before", yhat[0])
    # yhat = yhat + margin_mask
    # # print("after", yhat[0])
    # # print("after", one_hot_labels[0])
    # #targets = one_hot_labels.argmax(dim=1)
    
    # #scaling needed for 4 bit QNNs with ReLU 
    # #yhat = yhat*0.01
    # yhat = yhat*0.1
    # #yhat = torch.zeros(yhat.shape).cuda()
    # #yhat.zero_()
    # #print(yhat[0])
    # return torch.nn.functional.cross_entropy(yhat, y)
    
def margin_softmax_ce(yhat, y, m, epoch):
    # create one-hot tensor of ground truth
    #print("am usign msc")
    ###
    # some analysis on the NN outputs
    # print always the largest logits
    #values = yhat.max(dim=1).values
    #print("Largest logits", values)
    #print(yhat[0])
    ###
    #print("Epoch in mcel: ", epoch)
    #if epoch > 0:
    #print("Now applying margin")
    yhat = tanh_clamp_fixed(yhat, y, m, 100)
    one_hot_labels = torch.nn.functional.one_hot(y, yhat.shape[-1])
    margin_mask = -m * one_hot_labels
    #print("before", yhat[0])
    yhat = yhat + margin_mask
    # print("after", yhat[0])
    # print("after", one_hot_labels[0])
    #targets = one_hot_labels.argmax(dim=1)
    
    #scaling needed for 4 bit QNNs with ReLU 
    
    #scale logits by 1/m for normalization
    #yhat = yhat*0.01
    # if m > 0.1:
        # yhat = yhat/(m*torch.log2(torch.tensor([m]).cuda()))   

    # if m > 0.1:
        # yhat = yhat/(m)
    #yhat = torch.zeros(yhat.shape).cuda()
    #yhat.zero_()
    #print(yhat[0])
    return torch.nn.functional.cross_entropy(yhat, y)    
    
def tanh_clamp_fixed(logits, y=None, m=0.0, L=100.0):
    """
    Tanh-style clamping to a fixed range [-L, L], preserving actual margins.
    """
    # 1. smooth clamp
    logits_clamped = L * torch.tanh(logits / L)

    return logits_clamped
   
# def margin_softmax_ce(yhat, y, m, epoch):
    # # create one-hot tensor of ground truth
    # #print("am usign msc")
    # ###
    # # some analysis on the NN outputs
    # # print always the largest logits
    # #values = yhat.max(dim=1).values
    # #print("Largest logits", values)
    # #print(yhat[0])
    # ###
    # #print("Epoch in mcel: ", epoch)
    # #if epoch > 0:
    # #print("Now applying margin")
    # yhat = torch.clamp(yhat, min=-100, max=100)
    # one_hot_labels = torch.nn.functional.one_hot(y, yhat.shape[-1])
    # margin_mask = -m * one_hot_labels
    # #print("before", yhat[0])
    # yhat = yhat + margin_mask
    # # print("after", yhat[0])
    # # print("after", one_hot_labels[0])
    # #targets = one_hot_labels.argmax(dim=1)
    
    # #scaling needed for 4 bit QNNs with ReLU 
    # #yhat = yhat*0.01
    # yhat = yhat*0.1
    # #yhat = torch.zeros(yhat.shape).cuda()
    # #yhat.zero_()
    # #print(yhat[0])
    # return torch.nn.functional.cross_entropy(yhat, y)

# def binary_hingeloss(yhat, y, beta=10.0, eps=1e-8):
    # """
    # Probabilistic margin loss.
    
    # Encourages the probability of the correct class to be higher than all others,
    # with emphasis on small margins.
    
    # Args:
        # yhat : Tensor, shape (B, K) - logits
        # y : Tensor, shape (B,) - class indices
        # beta : float, sharpness parameter for soft-min emphasis
        # eps : float, small constant to avoid division by zero
    # Returns:
        # scalar loss
    # """
    # # Convert logits to probabilities (softmax)
    # probs = torch.nn.functional.softmax(yhat, dim=1)  # shape (B, K)

    # # Probabilities of the correct class
    # p_true = probs.gather(1, y.unsqueeze(1))  # shape (B, 1)

    # # Probabilities of all other classes
    # mask = torch.nn.functional.one_hot(y, yhat.shape[1]).bool()
    # p_other = probs.masked_fill(mask, 0.0)  # set true class to zero

    # # Margin between correct class probability and all others
    # margins = p_true - p_other  # shape (B, K)
    
    # # Remove zeros corresponding to true class
    # margins = margins[~mask]  # shape (B*(K-1),)

    # # Weight smaller margins more strongly using softmin
    # weights = 1.0 / (margins + eps)
    # weights = torch.clamp(weights, max=10.0)

    # # Softplus hinge on unit margin
    # loss = torch.nn.functional.softplus(1.0 - margins) * weights

    # return loss.mean()

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Clippy(torch.optim.Adam):
    def step(self, closure=None):
        loss = super(Clippy, self).step(closure=closure)
        for group in self.param_groups:
            for p in group['params']:
                p.data.clamp(-1,1)
        return loss

class Criterion:
    def __init__(self, method, name, param=None):
        self.method = method
        self.param = param
        self.name = name
        self.epoch = 0
    def applyCriterion(self, output, target):
        if self.param is not None:
            return self.method(output, target, self.param, self.epoch)
        else:
            return self.method(output, target)

def set_layer_mode(model, mode):
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if mode == "train":
                layer.training = True
            if mode == "eval":
                layer.eval = False

def parse_args(parser):
    parser.add_argument('--model', type=str, default=None,
                    help='VGG3/VGG7')
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
    parser.add_argument('--test-errors', type=int, default=1,
                        help='Test accuracy under errors')
    parser.add_argument('--margin-param', type=float, default=0,
                        help='Enter margin param')
    parser.add_argument('--verbosity', type=str, default="all",
                        help='How much info should be printed in the console. all/min')                    

def dump_exp_data(model, args, all_accuracies):
    to_dump = dict()
    to_dump["model"] = model.name
    # to_dump["method"] = model.method
    to_dump["batchsize"] = args.batch_size
    to_dump["epochs"] = args.epochs
    to_dump["learning_rate"] = args.lr
    to_dump["gamma"] = args.gamma
    to_dump["stepsize"] = args.step_size
    to_dump["traincrit"] = model.traincriterion.name
    to_dump["testcrit"] = model.testcriterion.name
    to_dump["test_error"] = all_accuracies
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
