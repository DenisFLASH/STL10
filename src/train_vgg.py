import torch.nn as nn
import torch.optim as optim
from torchvision import models

from tools import get_data_loaders
from transfer_learning import prepare_model, train_model, evaluate_model


BATCH_SIZE = 20
VALID_SPLIT = 0.2
SEED = 42

LR = 0.001
N_EPOCHS = 2

# Pre-trained VGG neural networks
MODELS_VGG = {
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "vgg19_bn": models.vgg19_bn(pretrained=True)
}

if __name__ == "__main__":

    # Load data
    loaders, classes = get_data_loaders(
        batch_size=BATCH_SIZE,
        valid_split=VALID_SPLIT,
        seed=SEED)
    train_loader, valid_loader, test_loader = loaders
    print(f"{len(train_loader)} train batches of size {BATCH_SIZE}")
    print(classes)

    for model_name, vgg in MODELS_VGG.items():
        print(f"\n\n{model_name.upper()}\n\n")

        # Freeze "feature" (conv+pool) layers, train all FC layers
        trainable_layers = vgg.classifier
        prepare_model(model=vgg,
                      trainable_layers=trainable_layers,
                      n_outputs=len(classes))

        # TODO remove tmp check
        for layer in [vgg.features, vgg.classifier]:
            print(layer)
            for p in layer.parameters():
                print(f"requires_grad: {p.requires_grad}")

        # Train model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(trainable_layers.parameters(), lr=LR)
        train_model(vgg, train_loader, criterion, optimizer, N_EPOCHS)

        # Evaluate model
        evaluate_model(vgg, test_loader, classes, criterion)
