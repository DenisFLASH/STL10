"""
Common tools for loading data from a local dataset.
"""

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def get_data_loaders(batch_size, valid_split, seed):
    transf = transforms.ToTensor()
    train_folder = datasets.ImageFolder(root="../dataset/train/",
                                        transform=transf)
    test_folder = datasets.ImageFolder(root="../dataset/test/",
                                       transform=transf)
    classes = test_folder.classes

    n_train = len(train_folder)
    indices = list(range(n_train))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(np.floor(valid_split * n_train))
    train_indices, valid_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(train_folder,
                              batch_size=batch_size,
                              sampler=train_sampler)

    valid_loader = DataLoader(train_folder,
                              batch_size=batch_size,
                              sampler=valid_sampler)

    test_loader = DataLoader(test_folder,
                             batch_size=batch_size)

    return (train_loader, valid_loader, test_loader), classes
