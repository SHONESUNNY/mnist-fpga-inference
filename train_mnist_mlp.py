"""Train a 784 -> 64 -> 10 MNIST MLP and classify custom 28x28 images.

Usage:
  python train_mnist_mlp.py                 # train, evaluate, optionally predict sent_image.npy
  python train_mnist_mlp.py --predict digit_output/sent_image.npy
  python train_mnist_mlp.py --predict digit_output/sent_image.bin

Pixel convention: 28x28 grayscale, black background and bright digit, uint8 [0,255].
The model input is scaled to float32 [0,1], matching torchvision MNIST ToTensor().
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DATA_DIR = Path("data")
OUTPUT_DIR = Path("mnist_artifacts")
MODEL_PATH = OUTPUT_DIR / "mnist_mlp_best.pth"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_sample_predictions.csv"


class MNISTMLP(nn.Module):
    """MLP: 784 inputs -> 64 hidden ReLU units -> 10 output logits."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 784)
        x = self.relu(self.fc1(x))
        # Return logits. CrossEntropyLoss applies the required log-softmax internally.
        return self.fc2(x)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("Evaluation loader was empty")
    return total_loss / total, 100.0 * correct / total


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    val_accuracy: float,
    path: Path,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": [784, 64, 10],
        "epoch": epoch,
        "validation_accuracy_percent": val_accuracy,
        "input_shape": [28, 28],
        "input_dtype": "uint8 source scaled to float32 [0,1]",
        "pixel_order": "row-major",
        "pixel_convention": "black background, bright digit",
    }
    torch.save(checkpoint, path)


def train() -> tuple[MNISTMLP, torch.device]:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ToTensor converts uint8 image pixels from [0,255] to float [0,1].
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )

    # Split only the official training set into train/validation. The official
    # test set is not examined until final evaluation after model selection.
    train_set, val_set = random_split(
        full_train,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / seen
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy_percent": val_accuracy,
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val accuracy {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            save_checkpoint(model, epoch, val_accuracy, MODEL_PATH)

    training_seconds = time.perf_counter() - training_start

    # Restore the checkpoint selected using validation data, then evaluate the
    # official test set exactly for the final report.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    metrics: dict[str, Any] = {
        "architecture": "784 -> 64 -> 10 (ReLU hidden layer)",
        "dataset": "MNIST",
        "training_examples": 55_000,
        "validation_examples": 5_000,
        "test_examples": len(test_set),
        "epochs_run": EPOCHS,
        "selected_epoch": best_epoch,
        "best_validation_accuracy_percent": round(best_val_accuracy, 4),
        "test_accuracy_percent": round(test_accuracy, 4),
        "test_loss": round(test_loss, 6),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "loss": "CrossEntropyLoss",
        "input_preprocessing": "grayscale uint8 [0,255] scaled to float32 [0,1]",
        "test_set_used_for_training": False,
        "training_seconds": round(training_seconds, 2),
        "history": history,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save example predictions from a fixed batch of official test images.
    model.eval()
    sample_images, sample_labels = next(iter(test_loader))
    with torch.inference_mode():
        sample_predictions = model(sample_images.to(device)).argmax(dim=1).cpu()
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "actual_label", "predicted_label", "correct"])
        for idx, (actual, predicted) in enumerate(zip(sample_labels.tolist(), sample_predictions.tolist())):
            writer.writerow([idx, actual, predicted, actual == predicted])

    print("\nFinal measured results")
    print(f"Selected epoch:        {best_epoch}")
    print(f"Best validation acc.:  {best_val_accuracy:.2f}%")
    print(f"Official test accuracy:{test_accuracy:.2f}%")
    print(f"Official test loss:    {test_loss:.4f}")
    print(f"Training time:         {training_seconds:.2f} s")
    print(f"Model saved to:        {MODEL_PATH}")
    print(f"Metrics saved to:      {METRICS_PATH}")
    print(f"Sample predictions:    {PREDICTIONS_PATH}")
    return model, device


def load_saved_model(device: torch.device) -> MNISTMLP:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained checkpoint at {MODEL_PATH}. Run without --predict first."
        )
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = MNISTMLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def read_custom_image(path: Path) -> np.ndarray:
    """Load a processed PNG, NPY array, or raw 784-byte BIN image."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".bin":
        raw = path.read_bytes()
        if len(raw) != 784:
            raise ValueError(f"Expected 784 bytes in {path}, got {len(raw)}")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(28, 28)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        with Image.open(path) as image:
            image = image.convert("L")
            if image.size != (28, 28):
                raise ValueError(
                    f"Expected an already-preprocessed 28x28 image; got {image.size}"
                )
            arr = np.asarray(image, dtype=np.uint8)
    else:
        raise ValueError("Supported formats: .npy, .bin, .png, .jpg, .jpeg, .bmp")

    arr = np.asarray(arr)
    if arr.size != 784:
        raise ValueError(f"Expected 784 pixel values, got {arr.size}")
    arr = arr.reshape(28, 28)

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Unsupported pixel dtype: {arr.dtype}")
    if np.min(arr) < 0 or np.max(arr) > 255:
        raise ValueError("Pixels must be in the [0,255] range")
    return arr.astype(np.uint8, copy=False)


def predict_custom_image(path: Path, model: MNISTMLP, device: torch.device) -> None:
    pixels = read_custom_image(path)
    # Match MNIST ToTensor(): same polarity and scale, [0,255] -> [0,1].
    tensor = torch.from_numpy(pixels.copy()).to(torch.float32).div_(255.0)
    tensor = tensor.reshape(1, 784).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted = int(probabilities.argmax().item())

    print(f"\nCustom image: {path}")
    print(f"Predicted digit: {predicted}")
    print(f"Model confidence (softmax): {probabilities[predicted].item() * 100:.2f}%")
    print("Class probabilities:")
    for digit, probability in enumerate(probabilities.tolist()):
        print(f"  {digit}: {probability * 100:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict",
        type=Path,
        help="Predict one processed 28x28 image (.npy, .bin, or 28x28 image file)",
    )
    args = parser.parse_args()

    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_saved_model(device)
        predict_custom_image(args.predict, model, device)
        return

    model, device = train()
    custom_path = Path("digit_output/sent_image.npy")
    if custom_path.exists():
        predict_custom_image(custom_path, model, device)
    else:
        print(
            "\nNo digit_output/sent_image.npy found. Draw and preprocess a digit, "
            "then run: python train_mnist_mlp.py --predict digit_output/sent_image.npy"
        )


if __name__ == "__main__":
    main()
