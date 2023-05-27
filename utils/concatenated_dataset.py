import numpy as np
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset


class CatChannelDataset(Dataset):
    """Modifies torchvision datasets to have extra identifying features as channels."""

    def __init__(
        self,
        n_channels,
        data,
        targets,
        transform,
        n_classes,
        is_test_data=False,
    ):
        self.n_channels = n_channels
        self.data = data
        self.targets = targets
        self.transform = transform
        self.n_classes = n_classes
        self.is_test_data = is_test_data

        # Organize data by classes.
        class_to_example = [[] for _ in range(self.n_classes)]
        for i in range(len(self.data)):
            key = self.targets[i]
            if isinstance(key, torch.Tensor):
                key = key.item()
            class_to_example[key].append(self.data[i])

        # Randomly pair up classes in the dataset.
        shuffled_classes = np.random.choice(np.arange(self.n_classes), self.n_classes, replace=False)
        self.class_mapping = {}
        for i in range(0, self.n_classes, 2):
            self.class_mapping[shuffled_classes[i]] = shuffled_classes[i + 1]
            self.class_mapping[shuffled_classes[i + 1]] = shuffled_classes[i]

        # Concatenate extra features to data.
        modified_data = []
        for i in range(len(self.data)):
            img, target = self.data[i], self.targets[i]

            # Get random image from different class to concatenate to current image.
            if isinstance(target, torch.Tensor):
                target = target.item()
            extra_target = self.class_mapping[target]
            extra_img = class_to_example[extra_target][np.random.randint(0, len(class_to_example[extra_target]))]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            if self.n_channels > 1:
                img = Image.fromarray(img)
                extra_img = Image.fromarray(extra_img)
            else:
                img = Image.fromarray(img.numpy(), mode="L")
                extra_img = Image.fromarray(extra_img.numpy(), mode="L")

            if self.transform is not None:
                img = self.transform(img)
                extra_img = self.transform(extra_img)

            if len(img.shape) < 3:
                # Add a dummy channel.
                img = img.unsqueeze(dim=0)
                extra_img = extra_img.unsqueeze(dim=0)
            
            if self.is_test_data:
                modified_data.append([img, torch.zeros(img.shape)])
            else:
                modified_data.append([img, extra_img])
        self.data = modified_data


    def __getitem__(self, index):
        imgs, target = self.data[index], self.targets[index]
        if self.is_test_data:
            return torch.cat(imgs, dim=0), target

        weight = torch.rand(1)
        return torch.cat([weight * imgs[0], (1 - weight) * imgs[1]], dim=0), target

    def __len__(self):
        return len(self.data)
