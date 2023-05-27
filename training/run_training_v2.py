import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from models.manifold_mixup import (
    ManifoldMixupDataset,
    ManifoldMixupModel,
    ManifoldMixupLoss,
)
from models.resnet import ResNet18
from torch.utils.data import DataLoader
from utils.concatenated_dataset import CatChannelDataset
from utils.data_utils import (
    load_mnist,
    load_fmnist,
    load_cifar10,
    load_cifar100,
    load_caltech256,
)
from utils.training_utils import full_train_test_loop_multi_train_loader
from utils.visualization_utils import plot_mixup_error

# Set up commandline arguments.
parser = argparse.ArgumentParser(description="Hyperparameters for model training.")
parser.add_argument("--task-name", dest="task_name", default="FMNIST", type=str)
parser.add_argument(
    "--model-type", dest="model_type", default="resnet", type=str
)
parser.add_argument("--optimizer", dest="optimizer", default="Adam", type=str)
parser.add_argument("--lr", dest="lr", default=1e-3, type=float)
parser.add_argument("--num-runs", dest="n_runs", default=1, type=int)
parser.add_argument("--subsample", dest="subsample", default=0, type=int)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    print("Device count: ", torch.cuda.device_count())
    print("GPU being used: {}".format(torch.cuda.get_device_name(0)))

# Fix seeds for reproducibility.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

filename_append = "regen_data"
perf_file = open(
    "runs_cat_channel/{}_{}_{}_{}_lr_{}_subsample_{}_runs_{}.out".format(
        args.task_name,
        args.model_type,
        args.optimizer,
        args.lr,
        args.subsample,
        args.n_runs,
        filename_append,
    ),
    "w",
)

out_dim = 10
n_channels = 3  # Number of channels in input image.
if args.task_name == "MNIST":
    n_channels = 1
    train_data, test_data = load_mnist()
if args.task_name == "FMNIST":
    n_channels = 1
    train_data, test_data = load_fmnist()
elif args.task_name == "CIFAR10":
    train_data, test_data = load_cifar10()
elif args.task_name == "CIFAR100":
    out_dim = 100
    train_data, test_data = load_cifar100()
elif args.task_name == "Caltech256":
    out_dim = 257  # There is an extra clutter class.
    train_data, test_data = load_caltech256()
else:
    sys.exit("Invalid task name.")

# Modify data.
train_datasets = []  # One for each run, not the most memory efficient way to do this but shrug.
for _ in range(args.n_runs):
    train_datasets.append(CatChannelDataset(
        n_channels=n_channels,
        data=train_data.data,
        targets=train_data.targets,
        transform=train_data.transform,
        n_classes=out_dim,
    ))
test_data = CatChannelDataset(
    n_channels=n_channels,
    data=test_data.data,
    targets=test_data.targets,
    transform=test_data.transform,
    n_classes=out_dim,
    is_test_data=True,
)

# Model/training parameters.
base_mixup_alpha = 1 # Baseline of uniform mixup.
mixup_alpha = 0  # Midpoint mixup.
relu_alpha = 11  # For polyrelu.
lr = args.lr
epochs = 100
batch_size = 750
num_runs = args.n_runs
test_interval = 1

print(
    "The following model/training parameters were used for this run: \n", file=perf_file
)
print("batch_size = ", batch_size, file=perf_file)
print("base_mixup_alpha = ", base_mixup_alpha, file=perf_file)
print("mixup_alpha = ", mixup_alpha, file=perf_file)
print("relu_alpha (only used for CNN) = ", relu_alpha, file=perf_file)
print("lr = ", lr, file=perf_file)
print("epochs = ", epochs, file=perf_file)
print("num_runs = ", num_runs, file=perf_file)
print("-------------------------------------------------\n", file=perf_file)

# Subsample as necessary.
if args.subsample > 0:
    for i in range(num_runs):
        train_datasets[i] = torch.utils.data.Subset(
            train_datasets[i],
            np.random.choice(
                list(range(len(train_datasets[i]))), size=args.subsample, replace=False
            ),
        )
    test_data = torch.utils.data.Subset(
        test_data,
        np.random.choice(
            list(range(len(test_data))), size=args.subsample, replace=False
        ),
    )

# Prepare data loaders.
base_dls, mixup_dls = [], []
for i in range(num_runs):
    base_dls.append(DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True))
    mixup_train = ManifoldMixupDataset(train_datasets[i], same_class_only=False, num_classes=out_dim, disclude_erm=False)
    mixup_dls.append(DataLoader(mixup_train, batch_size=batch_size, shuffle=True))
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Set up models.
base_model = ResNet18(num_classes=out_dim, n_channels=(n_channels * 2)).to(device)
base_mixup_model = ManifoldMixupModel(base_model, alpha=base_mixup_alpha).to(device)
mixup_model = ManifoldMixupModel(base_model, alpha=mixup_alpha).to(device)

criterion = torch.nn.CrossEntropyLoss()
mixup_criterion = ManifoldMixupLoss(criterion)

if args.optimizer == "SGD":
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=lr)
    base_mixup_optimizer = torch.optim.SGD(base_mixup_model.parameters(), lr=lr)
    mixup_optimizer = torch.optim.SGD(mixup_model.parameters(), lr=lr)
else:
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    base_mixup_optimizer = torch.optim.Adam(base_mixup_model.parameters(), lr=lr)
    mixup_optimizer = torch.optim.Adam(mixup_model.parameters(), lr=lr)


base_mixup_avg_errors, base_mixup_error_std = full_train_test_loop_multi_train_loader(
    model=base_mixup_model,
    test_loader=test_dl,
    test_loss_fn=criterion,
    train_loaders=mixup_dls,
    loss_fn=mixup_criterion,
    optimizer=base_mixup_optimizer,
    num_epochs=epochs,
    batch_size=batch_size,
    model_name="Uniform Mixup",
    out_file=perf_file,
    log_interval=0,
    device=device,
    num_runs=num_runs,
    test_interval=test_interval,
)

mixup_avg_errors, mixup_error_std = full_train_test_loop_multi_train_loader(
    model=mixup_model,
    test_loader=test_dl,
    test_loss_fn=criterion,
    train_loaders=mixup_dls,
    loss_fn=mixup_criterion,
    optimizer=mixup_optimizer,
    num_epochs=epochs,
    batch_size=batch_size,
    model_name="Midpoint Mixup",
    out_file=perf_file,
    log_interval=0,
    device=device,
    num_runs=num_runs,
    test_interval=test_interval,
)

base_avg_errors, base_error_std = full_train_test_loop_multi_train_loader(
    model=base_model,
    test_loader=test_dl,
    test_loss_fn=criterion,
    train_loaders=base_dls,
    loss_fn=criterion,
    optimizer=base_optimizer,
    num_epochs=epochs,
    batch_size=batch_size,
    model_name="Base",
    out_file=perf_file,
    log_interval=0,
    device=device,
    num_runs=num_runs,
    test_interval=test_interval,
)

# Create error plot.
plot_mixup_error(
    task=args.task_name,
    model_type=args.model_type,
    optimizer=args.optimizer,
    lr=args.lr,
    n_features=2,
    num_runs=num_runs,
    num_epochs=epochs,
    base_mixup_avg_errors=base_mixup_avg_errors,
    base_mixup_error_std=base_mixup_error_std,
    mixup_avg_errors=mixup_avg_errors,
    mixup_error_std=mixup_error_std,
    base_avg_errors=base_avg_errors,
    base_error_std=base_error_std,
    test_interval=test_interval,
    error_type="Test",
    filename_append=filename_append,
)
