import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from scipy.linalg import orth
from torch.utils.data import TensorDataset
from typing import List


def load_mnist():
    """Get MNIST dataset."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return (
        datasets.MNIST("data", train=True, download=True, transform=transform),
        datasets.MNIST("data", train=False, download=True, transform=transform),
    )


def load_fmnist():
    """Get FMNIST dataset."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return (
        datasets.FashionMNIST("data", train=True, download=True, transform=transform),
        datasets.FashionMNIST("data", train=False, download=True, transform=transform),
    )


def load_cifar10():
    """Get CIFAR-10 dataset."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return (
        datasets.CIFAR10("data", train=True, download=True, transform=transform),
        datasets.CIFAR10("data", train=False, download=True, transform=transform),
    )


def load_cifar100():
    """Get CIFAR-100 dataset."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    return (
        datasets.CIFAR100("data", train=True, download=True, transform=transform),
        datasets.CIFAR100("data", train=False, download=True, transform=transform),
    )


def load_caltech256():
    """Get Caltech-256 dataset."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    caltech_256 = datasets.Caltech256("data", download=True, transform=transform)
    return torch.utils.data.random_split(caltech_256, [len(caltech_256) - len(caltech_256) // 6, len(caltech_256) // 6])  # Mimic CIFAR split proportions.


def noisy_multi_feature_data(
    n_classes: int = 3,
    n_features: int = 2,
    n_patches: int = 9,
    data_height: int = 27,
    n_train: int = 1000,
    n_test: int = 100,
):
    """Multi-feature data with noise.

    Args:
        n_classes (int, optional): Number of classes. Defaults to 10.
        n_features (int, optional): Features associated with each class. Defaults to 2.
        n_patches (int, optional): Number of feature patches in each data point. Defaults to 9.
        data_height (int, optional): Height of each data point (so full dimension is data_height ** 2). Defaults to 27.
        n_train (int, optional): Number of training data points to generate. Defaults to 1000.
        n_test (int, optional): Number of test data points to generate. Defaults to 100.
    """
    # Hyperparameters from paper, but slightly modified for compute reasons.
    patch_height = int(data_height / np.sqrt(n_patches))
    patch_dim = patch_height**2
    signal_coeff_lb, signal_coeff_ub = 0.25, 0.75
    noise_coeff_scaling = 10 / (n_classes**1.5)

    # Sanity check hyperparameters.
    assert n_features % 2 == 0
    assert n_patches > n_features
    assert n_patches > n_features * n_classes
    assert np.abs(patch_dim * n_patches - (data_height**2)) < 1e-6

    # Associate orthonormal features with each class.
    all_features = orth(np.random.randn(patch_dim, n_features * n_classes)).T
    class_features = {
        k: all_features[(k * n_features) : ((k + 1) * n_features)].reshape(
            (n_features, patch_height, patch_height)
        )
        for k in range(n_classes)
    }

    def generate_noise_patch(y):
        noise_patch = np.zeros((patch_height, patch_height))
        for k in range(n_classes):
            if k != y:
                for ell in range(n_features):
                    noise_patch += (
                        np.random.uniform(signal_coeff_lb, signal_coeff_ub)
                        * noise_coeff_scaling
                        * class_features[k][ell]
                    )
        return noise_patch

    # Generate actual data.
    train_data, train_labels, test_data, test_labels = [], [], [], []
    for _ in range(n_train):
        y = np.random.randint(0, n_classes)
        cur_point = []
        # This corresponds to one signal patch per signal feature.
        for ell in range(n_features // 2):
            rand_coeff = np.random.uniform(
                signal_coeff_lb, signal_coeff_ub - signal_coeff_lb
            )
            cur_point.append(rand_coeff * class_features[y][ell])
            cur_point.append(
                (signal_coeff_ub - rand_coeff) * class_features[y][2 * ell + 1]
            )  # Mirrored feature.
        for _ in range(n_patches - n_features):
            cur_point.append(generate_noise_patch(y))
        train_data.append(np.array(cur_point).reshape((1, data_height, data_height)))
        train_labels.append(y)

    for _ in range(n_test):
        y = np.random.randint(0, n_classes)
        cur_point = []
        # We use one randomly selected signal feature for all of the signal patches.
        sig_feature = class_features[y][np.random.randint(0, n_features)]
        for ell in range(n_features):
            cur_point.append(
                np.random.uniform(signal_coeff_lb, signal_coeff_ub - signal_coeff_lb)
                * sig_feature
            )
        for _ in range(n_patches - n_features):
            cur_point.append(generate_noise_patch(y))
        test_data.append(np.array(cur_point).reshape((1, data_height, data_height)))
        test_labels.append(y)

    train_data, test_data = np.array(train_data), np.array(test_data)
    return TensorDataset(
        torch.FloatTensor(train_data), torch.LongTensor(train_labels)
    ), TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels))
