import torch.nn as nn


def replace_last_layer(model, n_outputs):
    """
    Replace the last fully connected linear layer by a new one,
    according to n_outputs.

    Returns
    -------
    None
        No output, since layer is replaced in place.
    """
    if model.__class__.__name__ == "VGG":
        print("replacing the last layer of VGG model")
        n_inputs = model.classifier[-1].in_features
        last_layer = nn.Linear(in_features=n_inputs, out_features=n_outputs)
        model.classifier[-1] = last_layer

    elif model.__class__.__name__ == "ResNet":
        print("replacing the last layer of ResNet model")
        n_inputs = model.fc.in_features
        last_layer = nn.Linear(in_features=n_inputs, out_features=n_outputs)
        model.fc = last_layer


def set_trainable_layers(model: nn.Module):
    """Prepare model to training:

    - Freeze the part of the network that will be used as feature extractor;
    - Set other part of the network as trainable.

    Returns
    -------
    trainable_layers : nn.Module
    """
    trainable_layers = None

    model.requires_grad_(False)

    if model.__class__.__name__ == "VGG":
        trainable_layers = model.classifier

    elif model.__class__.__name__ == "ResNet":
        trainable_layers = model.fc

    for p in trainable_layers.parameters():
        p.requires_grad = True

    return trainable_layers
