import sys

from torchvision import models

from tools import get_data_loaders, train_model
import transfer_learning as tl


BATCH_SIZE = 20
VALID_SPLIT = 0.2
SEED = 42


# Pre-trained neural networks
MODELS = {
    "vgg19_bn": models.vgg19_bn(pretrained=True),
    #"resnet50": models.resnet50(pretrained=True)
}

if __name__ == "__main__":

    EPOCHS = {
        "warmup": int(sys.argv[1]),
        "fine-tune": int(sys.argv[3])
    }
    LR = {
        "warmup": float(sys.argv[2]),
        "fine-tune": float(sys.argv[4])
    }
    print(f"params. Epochs: {EPOCHS}, learning rates: {LR}")

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
        tl.replace_last_layer(model=model, n_outputs=len(classes))
        tl.adapt_first_fc_layer(model=model)

        tl.freeze_feature_extractor(model=model)
        print(f"\nTraining FC layers (warm-up), freezing feature extractor\n")
        model = train_model(model=model,
                            train_loader=train_loader,
                            valid_loader=valid_loader,
                            lr=LR["warmup"],
                            n_epochs=EPOCHS["warmup"])

        # 2) Fine-tune the whole model
        model.requires_grad_(True)
        print(f"\nTraining the whole model (fine-tuning)\n")
        model = train_model(model=model,
                            train_loader=train_loader,
                            valid_loader=valid_loader,
                            lr=LR["fine-tune"],
                            n_epochs=EPOCHS["fine-tune"])
