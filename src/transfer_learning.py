import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import tools


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
    loaders, classes = tools.get_data_loaders(
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

    # Replace the last fully connected layer (which has index 6 for VGG)
    n_inputs = vgg16.classifier[6].in_features
    last_layer = nn.Linear(in_features=n_inputs,
                           out_features=len(classes))
    vgg16.classifier[6] = last_layer
    print(vgg16.classifier[6])

    if train_on_gpu:
        vgg16.cuda()

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=LR)

    # Train the model
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
            train_loss += batch_loss

        # average loss per epoch
        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}")
