from torchvision import models

from tools import get_data_loaders, train_model
from transfer_learning import replace_last_layer, freeze_feature_extractor


BATCH_SIZE = 20
VALID_SPLIT = 0.2
SEED = 42

LR = {"warmup": 0.01, "fine-tune": 0.001}
EPOCHS = {"warmup": 2, "fine-tune": 2}
# TODO pass as command line args

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
    train_loader, valid_loader, _ = loaders
    print(f"{len(train_loader)} train batches of size {BATCH_SIZE}")
    print(classes)

    for model_name, model in MODELS.items():
        print(f"\n\n{model_name.upper()}\n\n")

        # Two-stage Transfer Learning

        # 1) Replace last layer, freeze all feature layers, train FC layers
        replace_last_layer(model=model, n_outputs=len(classes))
        fc_layers = freeze_feature_extractor(model=model)
        train_model(model=model,
                    model_name=model_name,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    trainable_params=fc_layers.parameters(),
                    lr=LR["warmup"],
                    n_epochs=EPOCHS["warmup"])

        # 2) Fine-tune the whole model
        model.requires_grad_(True)
        train_model(model=model,
                    model_name=model_name,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    trainable_params=model.parameters(),
                    lr=LR["fine-tune"],
                    n_epochs=EPOCHS["fine-tune"])
