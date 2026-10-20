#!/usr/bin/env python3
"""Quantize the trained 784->64->10 MNIST MLP and validate an integer reference.

Place this file in the same directory as train_mnist_mlp.py, digit_output/, and
mnist_artifacts/. Run from any working directory:
    python phase4_quantize.py
    python phase4_quantize.py --predict digit_output/sent_image.npy

Quantization contract (also written to quantization_params.json):
- Input is uint8 pixels p in [0,255], representing real p/255 exactly.
- Weights use symmetric signed INT8, per-tensor scale max(abs(W))/127, zero-point 0.
- Layer 1 bias is INT32 in accumulator units for input_scale * weight_scale_1.
- Layer 1 accumulator is INT32. ReLU is followed by a defined integer requantization
  to UINT8 using a fixed-point multiplier and a right shift, rounded to nearest with
  ties away from zero, then saturated to [0,255].
- Layer 2 bias is INT32 in accumulator units for hidden_scale * weight_scale_2.
- Layer 2 outputs are signed INT32 accumulators. Since all ten scores share the same
  positive scale, argmax(accumulators) is the predicted class; no output requantizer
  is needed for classification.

The exporter writes two's-complement hex memories compatible with $readmemh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision import datasets


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "image_processing" else SCRIPT_DIR
DEFAULT_CHECKPOINT_CANDIDATES = [
    SCRIPT_DIR / "mnist_artifacts" / "mnist_mlp_best.pth",
    PROJECT_DIR / "mnist_artifacts" / "mnist_mlp_best.pth",
]
DEFAULT_DATA_DIR_CANDIDATES = [PROJECT_DIR / "data", SCRIPT_DIR / "data"]


class MNISTMLP(nn.Module):
    """Architecture must match train_mnist_mlp.py exactly."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 784)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def load_model(checkpoint_path: Path) -> tuple[MNISTMLP, dict[str, Any]]:
    if not checkpoint_path.exists():
        shown = "\n".join(f"  {p}" for p in DEFAULT_CHECKPOINT_CANDIDATES)
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}\n"
            f"Looked in these default locations:\n{shown}\n"
            "Make sure mnist_mlp_best.pth exists inside your mnist_artifacts folder."
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'. Is this the expected training checkpoint?")
    model = MNISTMLP().cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def symmetric_int8_quantize(weights: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(weights)))
    if not math.isfinite(max_abs) or max_abs == 0.0:
        # A zero tensor can be represented exactly by all-zero quantized weights.
        return np.zeros(weights.shape, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)
    return q, scale


def checked_int32(values: np.ndarray, what: str) -> np.ndarray:
    vals64 = np.asarray(values, dtype=np.int64)
    limits = np.iinfo(np.int32)
    if np.any(vals64 < limits.min) or np.any(vals64 > limits.max):
        raise OverflowError(f"{what} cannot be represented in signed INT32")
    return vals64.astype(np.int32)


def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    """Return (M, exponent), representing real_multiplier ~= M * 2**(exponent-31).

    M is a nonnegative signed-INT32 value. The exponent is kept separately so the
    RTL can use a signed 64-bit product followed by a specified shift.
    """
    if not math.isfinite(real_multiplier) or real_multiplier < 0.0:
        raise ValueError(f"Expected a finite nonnegative multiplier, got {real_multiplier}")
    if real_multiplier == 0.0:
        return 0, 0
    significand, exponent = math.frexp(real_multiplier)  # significand in [0.5, 1)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1
    if not 0 <= multiplier <= np.iinfo(np.int32).max:
        raise OverflowError("Quantized multiplier does not fit positive signed INT32")
    return multiplier, int(exponent)


def apply_quantized_multiplier(values: np.ndarray, multiplier: int, exponent: int) -> np.ndarray:
    """Bit-exact reference for round-nearest/ties-away-from-zero requantization.

    Real multiplier is approximately multiplier * 2**(exponent-31). The returned
    array is INT64 so saturation can be applied explicitly afterwards.
    """
    x = np.asarray(values, dtype=np.int64)
    product = x * np.int64(multiplier)
    right_shift = 31 - int(exponent)
    if right_shift > 0:
        half = np.int64(1 << (right_shift - 1))
        # Do not rely on NumPy's right-shift behavior for negative rounding.
        positive = (product + half) >> right_shift
        negative = -(((-product) + half) >> right_shift)
        return np.where(product >= 0, positive, negative).astype(np.int64)
    if right_shift == 0:
        return product
    # Left shift is unusual for this network, but defined for completeness.
    return product << (-right_shift)


def integer_infer_batch(
    pixels: np.ndarray,
    q_w1: np.ndarray,
    q_b1: np.ndarray,
    q_w2: np.ndarray,
    q_b2: np.ndarray,
    hidden_multiplier: int,
    hidden_exponent: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the specified integer model. Returns int32 scores, uint8 hidden, int64 acc1."""
    pix = np.asarray(pixels, dtype=np.uint8).reshape(-1, 784)
    # Use int64 as a checking/reference intermediate, then assert INT32 accumulator bounds.
    acc1_64 = pix.astype(np.int64) @ q_w1.astype(np.int64).T
    acc1_64 += q_b1.astype(np.int64)[None, :]
    lim = np.iinfo(np.int32)
    if np.any(acc1_64 < lim.min) or np.any(acc1_64 > lim.max):
        raise OverflowError("Layer 1 would overflow its specified signed INT32 accumulator")
    acc1 = acc1_64.astype(np.int32)

    # ReLU before requantization: nonpositive layer-1 accumulator maps to zero.
    nonnegative_acc1 = np.maximum(acc1.astype(np.int64), 0)
    q_hidden_wide = apply_quantized_multiplier(nonnegative_acc1, hidden_multiplier, hidden_exponent)
    q_hidden = np.clip(q_hidden_wide, 0, 255).astype(np.uint8)

    acc2_64 = q_hidden.astype(np.int64) @ q_w2.astype(np.int64).T
    acc2_64 += q_b2.astype(np.int64)[None, :]
    if np.any(acc2_64 < lim.min) or np.any(acc2_64 > lim.max):
        raise OverflowError("Layer 2 would overflow its specified signed INT32 accumulator")
    scores = acc2_64.astype(np.int32)
    return scores, q_hidden, acc1


def save_mem_hex(path: Path, values: np.ndarray, bits: int) -> None:
    """Write one unsigned hex token per line, encoding signed values as two's complement."""
    mask = (1 << bits) - 1
    width = bits // 4
    flat = np.asarray(values).reshape(-1)
    with path.open("w", encoding="ascii", newline="\n") as f:
        for value in flat:
            f.write(f"{int(value) & mask:0{width}x}\n")


def make_reference_from_export(params: dict[str, Any], out_dir: Path) -> dict[str, np.ndarray]:
    q_w1 = np.load(out_dir / "weights_fc1.npy").astype(np.int8)
    q_b1 = np.load(out_dir / "biases_fc1.npy").astype(np.int32)
    q_w2 = np.load(out_dir / "weights_fc2.npy").astype(np.int8)
    q_b2 = np.load(out_dir / "biases_fc2.npy").astype(np.int32)
    return {"q_w1": q_w1, "q_b1": q_b1, "q_w2": q_w2, "q_b2": q_b2}


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".bin":
        data = path.read_bytes()
        if len(data) != 784:
            raise ValueError(f"Expected 784 bytes in {path}; found {len(data)}")
        array = np.frombuffer(data, dtype=np.uint8)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        from PIL import Image
        with Image.open(path) as image:
            image = image.convert("L")
            if image.size != (28, 28):
                raise ValueError(f"Expected an already preprocessed 28x28 image, got {image.size}")
            array = np.asarray(image, dtype=np.uint8)
    else:
        raise ValueError("Supported image formats: .npy, .bin, .png, .jpg, .jpeg, .bmp")
    array = np.asarray(array)
    if array.size != 784 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Expected 784 numeric pixel values, got shape {array.shape}")
    if np.min(array) < 0 or np.max(array) > 255:
        raise ValueError("Pixel values must be in [0,255]")
    return array.astype(np.uint8, copy=False).reshape(28, 28)


def predict_saved_integer_model(image_path: Path, out_dir: Path) -> None:
    params_path = out_dir / "quantization_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Quantized parameters not found: {params_path}\nRun this script once without --predict first.")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    arrays = make_reference_from_export(params, out_dir)
    image = load_image(image_path)
    scores, hidden, _ = integer_infer_batch(
        image[None, ...], arrays["q_w1"], arrays["q_b1"], arrays["q_w2"], arrays["q_b2"],
        int(params["fc1"]["requant_multiplier"]), int(params["fc1"]["requant_exponent"]),
    )
    prediction = int(np.argmax(scores[0]))
    print(f"Image: {image_path}")
    print(f"Integer-reference prediction: {prediction}")
    print("Integer output scores (same positive scale across classes):")
    for digit, score in enumerate(scores[0].tolist()):
        print(f"  {digit}: {score}")
    print(f"Hidden activation nonzero count: {int(np.count_nonzero(hidden[0]))}/64")


def run_export(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else first_existing(DEFAULT_CHECKPOINT_CANDIDATES).resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else first_existing(DEFAULT_DATA_DIR_CANDIDATES).resolve()
    out_dir = args.output_dir.resolve() if args.output_dir else checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(checkpoint_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"MNIST data directory not found: {data_dir}. Pass --data-dir or restore your training data folder.")
    # Raw dataset tensors are uint8 [0,255], matching the training script's ToTensor preprocessing.
    calibration_set = datasets.MNIST(root=str(data_dir), train=True, download=False)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=False)
    cal_images = calibration_set.data
    if args.calibration_limit > 0:
        cal_images = cal_images[:args.calibration_limit]
    if len(cal_images) == 0:
        raise ValueError("Calibration set is empty")

    # Choose one unsigned 8-bit hidden-activation scale from actual floating-point ReLU activations.
    hidden_max = 0.0
    with torch.inference_mode():
        for start in range(0, len(cal_images), args.batch_size):
            batch = cal_images[start:start + args.batch_size].to(torch.float32).reshape(-1, 784) / 255.0
            hidden = model.relu(model.fc1(batch))
            hidden_max = max(hidden_max, float(hidden.max().item()))
    hidden_scale = hidden_max / 255.0 if hidden_max > 0.0 else 1.0

    w1 = model.fc1.weight.detach().cpu().numpy().astype(np.float64)
    b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float64)
    w2 = model.fc2.weight.detach().cpu().numpy().astype(np.float64)
    b2 = model.fc2.bias.detach().cpu().numpy().astype(np.float64)
    q_w1, w_scale1 = symmetric_int8_quantize(w1)
    q_w2, w_scale2 = symmetric_int8_quantize(w2)

    input_scale = 1.0 / 255.0
    b_scale1 = input_scale * w_scale1
    b_scale2 = hidden_scale * w_scale2
    q_b1 = checked_int32(np.rint(b1 / b_scale1), "fc1 quantized biases")
    q_b2 = checked_int32(np.rint(b2 / b_scale2), "fc2 quantized biases")

    # Layer-1 accumulator represents real preactivation in units input_scale * w_scale1.
    hidden_real_multiplier = (input_scale * w_scale1) / hidden_scale
    hidden_multiplier, hidden_exponent = quantize_multiplier(hidden_real_multiplier)

    # Validate the multiplier approximation and document its fixed-point interpretation.
    represented_multiplier = hidden_multiplier * (2.0 ** (hidden_exponent - 31))
    if hidden_real_multiplier != 0.0 and abs(represented_multiplier / hidden_real_multiplier - 1.0) > 2e-8:
        raise RuntimeError("Fixed-point requantization multiplier approximation unexpectedly inaccurate")

    # Export both binary NumPy arrays (for Python reference) and $readmemh files (for RTL).
    np.save(out_dir / "weights_fc1.npy", q_w1)
    np.save(out_dir / "biases_fc1.npy", q_b1)
    np.save(out_dir / "weights_fc2.npy", q_w2)
    np.save(out_dir / "biases_fc2.npy", q_b2)
    save_mem_hex(out_dir / "weights_fc1.mem", q_w1, 8)  # neuron-major: 64 rows of 784 weights
    save_mem_hex(out_dir / "biases_fc1.mem", q_b1, 32)
    save_mem_hex(out_dir / "weights_fc2.mem", q_w2, 8)  # neuron-major: 10 rows of 64 weights
    save_mem_hex(out_dir / "biases_fc2.mem", q_b2, 32)

    params: dict[str, Any] = {
        "format_version": 1,
        "architecture": [784, 64, 10],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input": {
            "dtype": "uint8", "shape": [28, 28], "scale": input_scale, "zero_point": 0,
            "real_value": "pixel_uint8 / 255.0", "pixel_order": "row-major", "polarity": "black=0, bright=255",
        },
        "quantization": {
            "weight_method": "symmetric per-tensor INT8, q=round(weight/scale), zero_point=0, range [-127,127]",
            "offline_rounding": "NumPy round-to-nearest-even for weights and biases",
            "runtime_requant_rounding": "round-to-nearest, ties away from zero",
            "accumulator_dtype": "signed int32; Python reference checks range before casting",
        },
        "fc1": {
            "weight_shape_out_in": [64, 784], "weight_dtype": "int8", "weight_scale": w_scale1,
            "bias_dtype": "int32", "bias_scale": b_scale1, "input_scale": input_scale,
            "accumulator_scale": b_scale1, "activation": "ReLU",
            "output_dtype": "uint8", "output_scale": hidden_scale, "output_zero_point": 0,
            "requant_real_multiplier": hidden_real_multiplier,
            "requant_multiplier": hidden_multiplier, "requant_exponent": hidden_exponent,
            "requant_right_shift": 31 - hidden_exponent,
            "requant_formula": "round_away_from_zero(accumulator * multiplier / 2**(31-exponent)); then saturate to [0,255]",
        },
        "fc2": {
            "weight_shape_out_in": [10, 64], "weight_dtype": "int8", "weight_scale": w_scale2,
            "bias_dtype": "int32", "bias_scale": b_scale2, "input_scale": hidden_scale,
            "accumulator_scale": b_scale2, "output_dtype": "int32", "output_scale": b_scale2,
            "output_requantization": "not required for argmax; all ten accumulators share the same positive scale",
        },
        "memory_layout": {
            "weights_fc1.mem": "64 output neurons, each with 784 consecutive input weights; row-major from PyTorch [out,in]",
            "biases_fc1.mem": "64 signed 32-bit two's-complement words, one per hidden neuron",
            "weights_fc2.mem": "10 output neurons, each with 64 consecutive hidden weights; row-major from PyTorch [out,in]",
            "biases_fc2.mem": "10 signed 32-bit two's-complement words, one per output class",
        },
        "calibration": {"dataset": "official MNIST training split", "samples_used": int(len(cal_images)), "max_float_relu_activation": hidden_max},
    }
    (out_dir / "quantization_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    # Compute the float and integer predictions for the full official 10,000-image test set.
    test_pixels = test_set.data.cpu().numpy().astype(np.uint8, copy=False)
    labels = test_set.targets.cpu().numpy().astype(np.uint8, copy=False)
    n_test = len(test_pixels)
    float_predictions = np.empty(n_test, dtype=np.uint8)
    integer_predictions = np.empty(n_test, dtype=np.uint8)
    integer_scores = np.empty((n_test, 10), dtype=np.int32)
    float_correct = 0
    int_correct = 0
    agree = 0
    arrays = {"q_w1": q_w1, "q_b1": q_b1, "q_w2": q_w2, "q_b2": q_b2}
    with torch.inference_mode():
        for start in range(0, n_test, args.batch_size):
            end = min(start + args.batch_size, n_test)
            batch_u8 = test_pixels[start:end]
            batch_float = torch.from_numpy(batch_u8.copy()).to(torch.float32).reshape(-1, 784) / 255.0
            pred_float = model(batch_float).argmax(dim=1).cpu().numpy().astype(np.uint8)
            scores_int, _, _ = integer_infer_batch(
                batch_u8, arrays["q_w1"], arrays["q_b1"], arrays["q_w2"], arrays["q_b2"], hidden_multiplier, hidden_exponent
            )
            pred_int = scores_int.argmax(axis=1).astype(np.uint8)
            float_predictions[start:end] = pred_float
            integer_predictions[start:end] = pred_int
            integer_scores[start:end] = scores_int
            float_correct += int(np.sum(pred_float == labels[start:end]))
            int_correct += int(np.sum(pred_int == labels[start:end]))
            agree += int(np.sum(pred_float == pred_int))

    np.savez_compressed(
        out_dir / "mnist_test_vectors.npz",
        images=test_pixels, labels=labels, float_predictions=float_predictions,
        integer_predictions=integer_predictions, integer_scores=integer_scores,
    )
    with (out_dir / "integer_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "actual_label", "float_prediction", "integer_prediction", "integer_matches_float", "integer_correct"])
        for i in range(n_test):
            writer.writerow([i, int(labels[i]), int(float_predictions[i]), int(integer_predictions[i]),
                             bool(float_predictions[i] == integer_predictions[i]), bool(integer_predictions[i] == labels[i])])

    # Export a compact set for an RTL testbench; files are one hex value per line.
    rtl_count = min(max(1, args.rtl_vectors), n_test)
    save_mem_hex(out_dir / "rtl_test_images.mem", test_pixels[:rtl_count].reshape(-1), 8)
    save_mem_hex(out_dir / "rtl_test_predictions.mem", integer_predictions[:rtl_count], 8)
    save_mem_hex(out_dir / "rtl_test_labels.mem", labels[:rtl_count], 8)
    (out_dir / "rtl_test_manifest.json").write_text(json.dumps({
        "number_of_images": rtl_count,
        "image_shape": [28, 28],
        "pixels_per_image": 784,
        "rtl_test_images.mem": "images concatenated consecutively, each 784 one-byte hex entries",
        "rtl_test_predictions.mem": "integer-reference expected class, one byte per image",
        "rtl_test_labels.mem": "MNIST ground-truth class, one byte per image",
    }, indent=2), encoding="utf-8")

    summary = {
        "checkpoint": str(checkpoint_path),
        "calibration_samples": int(len(cal_images)),
        "test_samples": int(n_test),
        "float_test_accuracy_percent": round(100.0 * float_correct / n_test, 4),
        "integer_test_accuracy_percent": round(100.0 * int_correct / n_test, 4),
        "integer_vs_float_prediction_agreement_percent": round(100.0 * agree / n_test, 4),
        "integer_correct_predictions": int(int_correct),
        "float_correct_predictions": int(float_correct),
        "matching_predictions": int(agree),
        "rtl_vectors_exported": int(rtl_count),
    }
    (out_dir / "quantization_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPhase 4 quantization complete")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Export directory: {out_dir}")
    print(f"Float test accuracy:              {summary['float_test_accuracy_percent']:.2f}%")
    print(f"Integer reference test accuracy:  {summary['integer_test_accuracy_percent']:.2f}%")
    print(f"Integer vs float prediction match:{summary['integer_vs_float_prediction_agreement_percent']:.2f}%")
    print(f"RTL test vectors exported:         {rtl_count}")
    print("\nImportant outputs:")
    for filename in [
        "quantization_params.json", "weights_fc1.mem", "biases_fc1.mem", "weights_fc2.mem", "biases_fc2.mem",
        "mnist_test_vectors.npz", "integer_test_predictions.csv", "rtl_test_images.mem",
        "rtl_test_predictions.mem", "rtl_test_labels.mem", "quantization_metrics.json",
    ]:
        print(f"  {out_dir / filename}")
    print("\nOnly proceed to RTL after reviewing quantization_metrics.json and confirming the integer accuracy is acceptable.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Path to mnist_mlp_best.pth")
    parser.add_argument("--data-dir", type=Path, help="Directory containing torchvision's MNIST folder")
    parser.add_argument("--output-dir", type=Path, help="Export directory (default: checkpoint's mnist_artifacts folder)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for calibration/evaluation (default: 256)")
    parser.add_argument("--calibration-limit", type=int, default=0, help="Use first N training images; 0 means all available training images")
    parser.add_argument("--rtl-vectors", type=int, default=100, help="Number of MNIST test images exported for RTL simulation (default: 100)")
    parser.add_argument("--predict", type=Path, help="Run just the exported integer reference on a custom .npy/.bin/28x28 image")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.calibration_limit < 0:
        parser.error("--calibration-limit must be >= 0")
    if args.rtl_vectors < 1:
        parser.error("--rtl-vectors must be >= 1")

    if args.predict:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else first_existing(DEFAULT_CHECKPOINT_CANDIDATES).resolve()
        out_dir = args.output_dir.resolve() if args.output_dir else checkpoint.parent
        predict_saved_integer_model(args.predict.resolve(), out_dir)
        return
    run_export(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
