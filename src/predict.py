
from pathlib import Path

from tools import get_data_loaders, evaluate_model


BATCH_SIZE = 20
VALID_SPLIT = 0.2
SEED = 42

# relative path to the current file
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

if __name__ == "__main__":

    # Load data
    loaders, classes = get_data_loaders(
        batch_size=BATCH_SIZE,
        valid_split=VALID_SPLIT,
        seed=SEED)
    _, _, test_loader = loaders

    filename = "vgg19_bn.pt"

    model_path = MODEL_DIR / filename
    evaluate_model(model_path,
                   test_loader,
                   classes)
