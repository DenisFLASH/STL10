import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn


train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("CUDA is available!!!  Training on GPU ...")
else:
    print("CUDA is not available.  Training on CPU ...")


def prepare_model(model,
                  trainable_layers,
                  n_outputs):
    """Prepare model to training:

    - Replace the last FC layer by a new one, according to n_outputs;
    - Freeze the part of the network that will be used as feature extractor;
    - Set other part of the network as trainable.

    Returns
    -------
    None
        No return, since choosing which part to freeze or train
        is applied in place.
    """
    # Freeze all layers, then set trainable layers to train
    model.requires_grad_(False)
    for p in trainable_layers.parameters():
        p.requires_grad = True

    # Replace the last fully connected layer
    n_inputs = model.classifier[-1].in_features
    last_layer = nn.Linear(in_features=n_inputs,
                           out_features=n_outputs)
    model.classifier[-1] = last_layer


def train_model(model,
                train_loader,
                criterion,
                optimizer,
                n_epochs):

    if train_on_gpu:
        model.cuda()

    print(f"\nTraining the model\n")
    model.train()

    for epoch in range(n_epochs):
        print(f"epoch {epoch+1}")

        train_loss = 0.0

        for i, (data, target) in enumerate(train_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()  # clear all gradients
            output = model(data)  # forward pass: predict
            loss = criterion(output, target)
            loss.backward()  # backward pass: compute gradient of the loss
            optimizer.step()  # update parameters

            batch_loss = loss.item() * data.size(0)
            if (i + 1) % 100 == 0:
                print(f"batch: {i+1}\tTraining loss per example: {loss.item()}")
            train_loss += batch_loss

        # average loss per epoch
        train_loss /= len(train_loader.dataset)
        print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}")
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
    for i, (data, target) in enumerate(test_loader):

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        if (i + 1) % 100 == 0:
            print(f"test batch {i + 1}/{len(test_loader)}, loss {loss.item()}")

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
