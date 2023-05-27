#!/bin/bash

# FMNIST experiments.
sbatch training/run_training_v2.py --task-name=FMNIST --model-type=resnet --optimizer=Adam --subsample=0 --num-runs=3
sbatch training/run_training_v2.py --task-name=FMNIST --model-type=resnet --optimizer=Adam --subsample=0 --num-runs=5

# CIFAR10 experiments.
sbatch training/run_training_v2.py --task-name=CIFAR10 --model-type=resnet --optimizer=Adam --subsample=0 --num-runs=3
sbatch training/run_training_v2.py --task-name=CIFAR10 --model-type=resnet --optimizer=Adam --subsample=0 --num-runs=5

# CIFAR100 experiments.
sbatch training/run_training_v2.py --task-name=CIFAR100 --model-type=resnet --optimizer=Adam --subsample=0 --num-runs=3
sbatch training/run_training_v2.py --task-name=CIFAR100 --model-type=resnet --optimizer=Adam --subsample=0 --num-runs=5

