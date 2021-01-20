"""
Common tools for loading data from a local dataset.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("CUDA is available!!!  Training on GPU ...")
else:
    print("CUDA is not available.  Training on CPU ...")


def get_data_loaders(batch_size, valid_split, seed):
    transf = transforms.ToTensor()
    # TODO crop, normalize (see Pytorch doc)
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


def train_model(model,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                n_epochs):

    print(f"\nTraining the model\n")

    if train_on_gpu:
        model.cuda()

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}")

        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()

        for i, (data, target) in enumerate(train_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()  # clear all gradients
            output = model(data)  # forward pass: predict
            loss = criterion(output, target)  # average loss per item
            train_loss += loss.item()

            loss.backward()  # backward pass: compute gradient of the loss
            optimizer.step()  # update parameters


        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()

        print(f"Train loss: {train_loss:.6f} \tValid loss: {valid_loss:.6f}")

        # average loss per epoch
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        print(f"Train loss: {train_loss:.6f} \tValid loss: {valid_loss:.6f}")

        # TODO return losses_train, losses_valid; plot in notebook


def evaluate_model(model,
                   test_loader,
                   classes,
                   criterion):
    """
    Evaluate model's prediction quality.
    """
    test_loss = 0.0
    all_preds = np.array([], "int")
    all_targets = np.array([], "int")

    model.eval()  # eval mode

    # iterate over test data
    for data, target in test_loader:

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        if train_on_gpu:
            pred, target = pred.cpu(), target.cpu()

        all_preds = np.concatenate((all_preds, pred.numpy()))
        all_targets = np.concatenate((all_targets, target.numpy()))

    test_loss /= len(test_loader.dataset)
    print(f"Avg test Loss: {test_loss:.6f}\n")

    cm = confusion_matrix(y_true=all_targets, y_pred=all_preds)
    n_total = cm.sum()
    n_correct = cm.diagonal().sum()
    acc_total = 100 * n_correct / n_total
    print(f"Total test accuracy: {acc_total:.1f}% ({n_correct}/{n_total})\n")

    for i, cl in enumerate(classes):
        tp = cm[i, i]
        n = cm[i].sum()
        acc = 100 * tp / n
        print(f"Test accuracy of {cl}:\t {acc:.0f}% ({tp}/{n})")

    df_cm = pd.DataFrame(cm,
                         columns=classes,
                         index=[c.upper() for c in classes])
    print(f"\nconfusion_matrix (TRUE label is uppercase):\n\n{df_cm}")
