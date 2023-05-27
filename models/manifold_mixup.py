# Forked from: https://github.com/shivamsaboo17/ManifoldMixup.
# All credit should go to original author; this is only a minor modification of the above.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import copy
import warnings


class ManifoldMixupDataset(Dataset):
    def __init__(
        self, dataset, same_class_only=False, num_classes=10, disclude_erm=False
    ):
        self.dataset = dataset
        self.same_class_only = same_class_only
        self.disclude_erm = disclude_erm
        if same_class_only:
            self.class_to_example = [[] for i in range(num_classes)]
            for example, label in dataset:
                self.class_to_example[label].append((example, label))

    def __getitem__(self, index):
        x_0, y_0 = self.dataset[index]
        if self.same_class_only:
            new_idx = np.random.randint(0, len(self.class_to_example[y_0]))
            x_1, y_1 = self.class_to_example[y_0][new_idx]
        else:
            new_idx = np.random.randint(0, len(self.dataset))
            if self.disclude_erm:
                # Don't mix with same point.
                while new_idx == index:
                    # This is faster than diff1d.
                    new_idx = np.random.randint(0, len(self.dataset))
            x_1, y_1 = self.dataset[new_idx]
        return [x_0, x_1], [y_0, y_1]

    def __len__(self):
        return len(self.dataset)


class ManifoldMixupModel(nn.Module):
    def __init__(
        self,
        model,
        alpha=1.0,
        interpolation_adv=False,
        mixup_all=False,
        use_input_mixup=True,
    ):
        super(ManifoldMixupModel, self).__init__()
        self.use_input_mixup = use_input_mixup
        self.model = copy.deepcopy(model)
        if not mixup_all:
            self.module_list = list(
                filter(
                    lambda module: isinstance(module, MixupModule),
                    list(self.model.modules()),
                )
            )
        else:
            self.module_list = list(self.model.modules())
        print(f"{len(self.module_list)} modules eligible for mixup.")
        self.alpha = alpha
        self.intermediate_other = None
        self.lam = None
        self.interpolation_adv = interpolation_adv
        self._hooked = None
        self._warning_raised = False

    def forward(self, x, switch_adv=False):
        if not self.training or (
            not self.use_input_mixup and len(self.module_list) == 0
        ):
            # Don't perform mix-up if evaluating or all mix-up is turned off.
            return self.model(x)

        x_0, x_1 = x
        # Generate a mixing param per batch.
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 0.5

        l_l = -1 if self.use_input_mixup else 0
        k = np.random.randint(l_l, len(self.module_list))
        if k == -1:
            x_ = self.lam * x_0 + (1 - self.lam) * x_1
            out = self.model(x_)
        else:
            self._update_hooked(False)
            fetcher_hook = self.module_list[k].register_forward_hook(self.hook_fetch)
            self.model(x_1)
            fetcher_hook.remove()
            self._update_hooked(False)
            modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
            out = self.model(x_0)
            modifier_hook.remove()
        self._update_hooked(None)
        if self.interpolation_adv and not switch_adv:
            return out, x, self, self.lam
        return out, self.lam

    def hook_modify(self, module, input, output):
        if not self._hooked:
            output = (1 - self.lam) * self.intermediate_other + self.lam * output
            self._update_hooked(True)
            return output

    def hook_fetch(self, module, input, output):
        if not self._hooked:
            self.intermediate_other = output
            self._update_hooked(True)
        else:
            if not self._warning_raised:
                warnings.warn(
                    "One of the mixup modules defined in the model is used more than once in forward pass. Mixup will happen only at first call.",
                    Warning,
                )
                self._warning_raised = True

    def _update_hooked(self, flag):
        self._hooked = flag


class ManifoldMixupLoss(nn.Module):
    def __init__(self, criterion):
        super(ManifoldMixupLoss, self).__init__()
        self.criterion = criterion

    def forward(self, outs, y):
        out, lam = outs
        y_0, y_1 = y
        # This only works for cross-entropy/NLL.
        loss_0, loss_1 = self.criterion(out, y_0), self.criterion(out, y_1)
        return lam * loss_0 + (1 - lam) * loss_1


class MixupModule(nn.Module):
    def __init__(self, module):
        super(MixupModule, self).__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)
