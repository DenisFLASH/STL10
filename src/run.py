import torch.nn as nn
import torch.optim as optim
from torchvision import models

from tools import get_data_loaders, train_model, evaluate_model
from transfer_learning import replace_last_layer, set_trainable_layers


BATCH_SIZE = 20
VALID_SPLIT = 0.2
SEED = 42

LR = 0.01
N_EPOCHS = 2

# Pre-trained VGG neural networks
MODELS = {
    "vgg19_bn": models.vgg19_bn(pretrained=True),
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

    for model_name, model in MODELS.items():
        print(f"\n\n{model_name.upper()}\n\n")

        replace_last_layer(model=model,
                           n_outputs=len(classes))
        trainable_layers = set_trainable_layers(model=model)

        # Train model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(trainable_layers.parameters(), lr=LR)
        path = train_model(model,
                           model_name,
                           train_loader,
                           valid_loader,
                           criterion,
                           optimizer,
                           N_EPOCHS)

        # Evaluate model
        evaluate_model(model,
                       path,
                       test_loader,
                       classes,
                       criterion)
