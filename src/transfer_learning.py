import numpy as np
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
            if (i + 1) % 10 == 0:
                print(f"batch: {i+1}\tTraining loss per example: {loss.item()}")
            train_loss += batch_loss

        # average loss per epoch
        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}")
        # TODO return losses_train, losses_valid; plot in notebook


def evaluate_model(model,
                   test_loader,
                   classes,
                   criterion,
                   batch_size):
    """
    Evaluate model's prediction quality.
    """
    test_loss = 0.0
    class_correct = [0. * len(classes)]
    class_total = [0. * len(classes)]

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

        if (i + 1) % 10 == 0:
            print(f"test batch {i + 1}/{len(test_loader)}, loss {loss.item()}")

        # update  test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(
            correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
            correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            # TODO replace 'i' or vectorize the calculation (confusion matrix?)
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}\n")

    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (
            classes[i]))
    # TODO confusion matrix
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

