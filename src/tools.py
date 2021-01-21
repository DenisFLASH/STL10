"""
Common tools for:
 - loading data from a local dataset;
 - model training and evaluation.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm


train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("CUDA is available!!! Using GPU ...")
else:
    print("CUDA is not available. Using CPU ...")

# relative path to the current file
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def get_data_loaders(batch_size, valid_split, seed):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(  # the same normalization as for ImageNet
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

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
                trainable_params,
                lr,
                n_epochs):

    if train_on_gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trainable_params, lr=lr)
    print(f"Learning rate: {lr}")

    valid_loss_min = np.Inf  # track change in validation loss to save the model
    best_model = None
    losses_train, losses_val = [], []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}")

        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()

        for data, target in tqdm(train_loader):
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
        for data, target in tqdm(valid_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()

        # average loss per epoch
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        losses_train.append(train_loss)
        losses_val.append(valid_loss)

        print(f"Train loss: {train_loss:.6f} \tValid loss: {valid_loss:.6f}")

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            filename = f"{model.__class__.__name__}_lr{lr}_ep{epoch+1}.pt"
            path = MODEL_DIR / filename
            print(f"Saving model to {path}")
            torch.save(model, path)
            valid_loss_min = valid_loss
            best_model = copy.deepcopy(model)

    print(f"train losses per epoch:\n{losses_train}")
    print(f"valid losses per epoch:\n{losses_val}")

    return best_model


def evaluate_model(model_path,
                   test_loader,
                   classes):
    """
    Evaluate model's prediction quality.
    """
    print(f"Load the model from {model_path}")
    model = torch.load(model_path)

    test_loss = 0.0
    all_preds = np.array([], "int")
    all_targets = np.array([], "int")

    criterion = nn.CrossEntropyLoss()

    model.eval()  # eval mode

    # iterate over test data
    for data, target in tqdm(test_loader):

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
    print(f"\nTest Loss: {test_loss:.6f}\n")

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
