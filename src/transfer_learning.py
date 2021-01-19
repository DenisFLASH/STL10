import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tools import get_data_loaders


BATCH_SIZE = 20
VALID_SPLIT = 0.2
SEED = 42

LR = 0.001
N_EPOCHS = 2

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("CUDA is available!!!  Training on GPU ...")
else:
    print("CUDA is not available.  Training on CPU ...")


if __name__ == "__main__":

    # Load data
    loaders, classes = get_data_loaders(
        batch_size=BATCH_SIZE,
        valid_split=VALID_SPLIT,
        seed=SEED)
    train_loader, valid_loader, test_loader = loaders
    print(f"{len(train_loader)} train batches of size {BATCH_SIZE}")
    print(classes)

    # Instantiate a pre-trained VGG16 neural network
    vgg16 = torchvision.models.vgg16(pretrained=True)
    print(vgg16)

    # Freeze weights for all "features" (convolution and pooling) layers
    for param in vgg16.features.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    n_inputs = vgg16.classifier[-1].in_features
    last_layer = nn.Linear(in_features=n_inputs,
                           out_features=len(classes))
    vgg16.classifier[-1] = last_layer

    # TODO remove tmp check
    for cl in vgg16.classifier:
        print(cl)
        for p in cl.parameters():
            print(f"requires_grad: {p.requires_grad}")

    if train_on_gpu:
        vgg16.cuda()

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=LR)

    #############
    ### TRAIN ###
    #############
    vgg16.train()

    for epoch in range(N_EPOCHS):

        train_loss = 0.0

        for i, (data, target) in enumerate(train_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()  # clear all gradients
            output = vgg16(data)  # forward pass: predict
            loss = criterion(output, target)
            loss.backward()  # backward pass: compute gradient of the loss
            optimizer.step()  # update parameters

            batch_loss = loss.item() * data.size(0)
            if (i + 1) % 10 == 0:
                print(
                    f"batch: {i+1}\t"
                    f"Training loss per example: {batch_loss/BATCH_SIZE}")
                # TODO remove batch_size, use loss.item()
            train_loss += batch_loss

        # average loss per epoch
        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}")
        # TODO return losses_train, losses_valid; plot in notebook

    ################
    ### EVALUATE ###
    ################
    test_loss = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    vgg16.eval()  # eval mode

    # iterate over test data
    for i, (data, target) in enumerate(test_loader):

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg16(data)
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
        for i in range(BATCH_SIZE):
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
