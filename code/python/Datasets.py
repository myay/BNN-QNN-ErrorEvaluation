"""Datasets and data loaders for run_master.py.

Transforms (unchanged from the former run scripts):
  MNIST / FMNIST / KMNIST / SVHN : ToTensor() for train and test
  CIFAR10 / CIFAR100             : RandomCrop(32, 4) + RandomHorizontalFlip + Normalize(0.5) for
                                   train, Normalize(0.5) for test
  IMAGENETTE                     : Resize(size) + RandomHorizontalFlip + ImageNet Normalize for
                                   BOTH train and test.
                                   KNOWN ISSUE: the random flip on the test set makes every test
                                   pass (and every point of the bit-error sweep) see different
                                   images. Kept on purpose so results stay comparable with the
                                   former scripts; to be fixed separately.
"""
import os

import torch
from torchvision import datasets, transforms

# name -> (in_channels, input_size, num_classes)
DATASET_INFO = {
    "MNIST": (1, 28, 10),
    "FMNIST": (1, 28, 10),
    "KMNIST": (1, 28, 10),
    "SVHN": (3, 32, 10),
    "CIFAR10": (3, 32, 10),
    "CIFAR100": (3, 32, 100),
    "IMAGENETTE": (3, 64, 10),
}
DATASET_NAMES = tuple(DATASET_INFO)

# accepted spellings -> canonical name
_ALIASES = {
    "FASHIONMNIST": "FMNIST", "FASHION": "FMNIST", "FMNIST": "FMNIST",
    "MNIST": "MNIST", "KMNIST": "KMNIST", "SVHN": "SVHN",
    "CIFAR10": "CIFAR10", "CIFAR-10": "CIFAR10", "CIFAR100": "CIFAR100", "CIFAR-100": "CIFAR100",
    "IMAGENETTE": "IMAGENETTE", "IMAGENETTE2": "IMAGENETTE",
}


def canonical_dataset(name):
    key = str(name).upper()
    if key not in _ALIASES:
        raise ValueError("Unknown dataset {!r}. Valid datasets: {}".format(name, ", ".join(DATASET_NAMES)))
    return _ALIASES[key]


def dataset_geometry(name, image_size=None):
    """(in_channels, input_size, num_classes) of a dataset."""
    name = canonical_dataset(name)
    c, s, n = DATASET_INFO[name]
    if name == "IMAGENETTE" and image_size is not None:
        s = image_size
    return c, s, n


def get_datasets(name, data_dir="data", image_size=None, download=True):
    """Return (train_set, test_set) for a dataset."""
    name = canonical_dataset(name)
    plain = transforms.Compose([transforms.ToTensor()])

    if name == "MNIST":
        return (datasets.MNIST(data_dir, train=True, download=download, transform=plain),
                datasets.MNIST(data_dir, train=False, transform=plain))
    if name == "FMNIST":
        return (datasets.FashionMNIST(data_dir, train=True, download=download, transform=plain),
                datasets.FashionMNIST(data_dir, train=False, transform=plain))
    if name == "KMNIST":
        root = os.path.join(data_dir, "KMNIST")
        return (datasets.KMNIST(root, train=True, download=download, transform=plain),
                datasets.KMNIST(root, train=False, download=download, transform=plain))
    if name == "SVHN":
        root = os.path.join(data_dir, "SVHN")
        return (datasets.SVHN(root=root, split="train", download=download, transform=plain),
                datasets.SVHN(root=root, split="test", download=download, transform=plain))
    if name in ("CIFAR10", "CIFAR100"):
        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), norm])
        t_test = transforms.Compose([transforms.ToTensor(), norm])
        cls = datasets.CIFAR10 if name == "CIFAR10" else datasets.CIFAR100
        return (cls(data_dir, train=True, download=download, transform=t_train),
                cls(data_dir, train=False, transform=t_test))
    if name == "IMAGENETTE":
        size = image_size if image_size is not None else DATASET_INFO["IMAGENETTE"][1]
        t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        root = os.path.join(data_dir, "imagenette2")
        for split in ("train", "val"):
            if not os.path.isdir(os.path.join(root, split)):
                raise IOError("ImageNette not found at {} (expected {}/train and {}/val; "
                              "no automatic download)".format(root, root, root))
        return (datasets.ImageFolder(os.path.join(root, "train"), transform=t),
                datasets.ImageFolder(os.path.join(root, "val"), transform=t))
    raise ValueError("Unknown dataset {!r}".format(name))


def get_loaders(train_set, test_set, batch_size, test_batch_size, num_workers=1):
    """(train_loader, test_loader). The kernels need a GPU, so memory is always pinned.

    The test loader is shuffled too. That does not change the accuracy, but it
    consumes random numbers; it is kept so that a given --seed gives the same
    run as with the former scripts.
    """
    kwargs = {"num_workers": num_workers, "pin_memory": True, "shuffle": True}
    return (torch.utils.data.DataLoader(train_set, batch_size=batch_size, **kwargs),
            torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, **kwargs))
