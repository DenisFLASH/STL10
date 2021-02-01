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


def freeze_feature_extractor(model: nn.Module):
    """Prepare model to training:

    - Freeze the part of the network that will be used as feature extractor;
    - Set other part of the network as trainable.

    Returns
    -------
    None
    """
    fc_layers = None

    model.requires_grad_(False)

    if model.__class__.__name__ == "VGG":
        fc_layers = model.classifier

    elif model.__class__.__name__ == "ResNet":
        fc_layers = model.fc

    for p in fc_layers.parameters():
        p.requires_grad = True


# TODO drop this function, after resizing initial input to 224*224
def adapt_first_fc_layer(model: nn.Module):
    """
    Since our image starts from size 96*96 (and not 224*224 as ImageNet),
    after passing through conv+pool layers it becomes 3*3*512 instead of 7*7*512.
    So, we have to adapt the way the first FC layer is connected to the last
    pooling layer.
    """
    if model.__class__.__name__ == "VGG":

        first_fc_in_features = 3 * 3 * 512
        first_fc_out_features = model.classifier[0].out_features
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        model.classifier[0] = nn.Linear(in_features=first_fc_in_features,
                                        out_features=first_fc_out_features)
