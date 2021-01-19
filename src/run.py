import torch.nn as nn
import torch.optim as optim
from torchvision import models

from tools import get_data_loaders
from transfer_learning import replace_last_layer, set_trainable_layers, train_model, evaluate_model


BATCH_SIZE = 20
VALID_SPLIT = 0.2
SEED = 42

LR = 0.001
N_EPOCHS = 2

# Pre-trained VGG neural networks
MODELS = {
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "vgg19_bn": models.vgg19_bn(pretrained=True),
    "resnet18": models.resnet18(pretrained=True),
    "resnet34": models.resnet34(pretrained=True),
    "resnet50": models.resnet50(pretrained=True)
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

    for model_name, vgg in MODELS.items():
        print(f"\n\n{model_name.upper()}\n\n")

        replace_last_layer(model=vgg,
                           n_outputs=len(classes))
        trainable_layers = set_trainable_layers(model=vgg)

        # Train model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(trainable_layers.parameters(), lr=LR)
        train_model(vgg, train_loader, criterion, optimizer, N_EPOCHS)

        # Evaluate model
        evaluate_model(vgg, test_loader, classes, criterion)
