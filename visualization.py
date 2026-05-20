import argparse
from pathlib import Path
from struct import unpack
import gzip
import pickle

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import mynn as nn


ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs"


def load_test_data():
    test_images_path = ROOT / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz"
    test_labels_path = ROOT / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz"

    with gzip.open(test_images_path, "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(test_labels_path, "rb") as f:
        _, num = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return test_imgs.astype(np.float64) / 255.0, test_labs


def load_model(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    model_type = state.get("model_type") if isinstance(state, dict) else "MLP"
    model = nn.models.Model_CNN() if model_type == "CNN" else nn.models.Model_MLP()
    model.load_model(path)
    return model


def confusion_matrix(pred, labels, n=10):
    mat = np.zeros((n, n), dtype=int)
    for y, p in zip(labels, pred):
        mat[int(y), int(p)] += 1
    return mat


def _font(size=14):
    for name in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def plot_confusion(mat):
    cell = 42
    margin = 70
    width = margin + cell * 10 + 25
    height = margin + cell * 10 + 45
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(12)
    title_font = _font(16)
    draw.text((margin, 15), "Confusion matrix", fill=(25, 25, 25), font=title_font)
    max_val = max(1, int(mat.max()))
    for i in range(10):
        draw.text((margin - 28, margin + i * cell + 12), str(i), fill=(40, 40, 40), font=font)
        draw.text((margin + i * cell + 16, margin - 25), str(i), fill=(40, 40, 40), font=font)
        for j in range(10):
            val = int(mat[i, j])
            shade = 255 - int(190 * val / max_val)
            color = (shade, shade + min(255 - shade, 25), 255)
            x0 = margin + j * cell
            y0 = margin + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(230, 230, 230))
            if val:
                draw.text((x0 + 11, y0 + 13), str(val), fill=(20, 20, 20), font=font)
    draw.text((margin + 145, height - 25), "Predicted", fill=(40, 40, 40), font=font)
    draw.text((8, margin + 175), "True", fill=(40, 40, 40), font=font)
    img.save(FIGS / "confusion_matrix.png")


def _mnist_image(arr, scale=4):
    data = np.clip(arr.reshape(28, 28) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(data, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST).convert("RGB")


def plot_misclassified_examples(model, test_imgs, test_labs):
    pred = np.argmax(model(test_imgs), axis=1)
    wrong = np.where(pred != test_labs)[0][:12]
    if wrong.size == 0:
        wrong = np.arange(min(12, test_imgs.shape[0]))

    scale = 4
    tile = 28 * scale
    label_h = 24
    cols, rows = 4, 3
    img = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(img)
    font = _font(13)
    for n, idx in enumerate(wrong[:rows * cols]):
        x = (n % cols) * tile
        y0 = (n // cols) * (tile + label_h)
        img.paste(_mnist_image(test_imgs[idx], scale), (x, y0))
        draw.text((x + 5, y0 + tile + 4), f"t:{int(test_labs[idx])} p:{int(pred[idx])}", fill=(20, 20, 20), font=font)
    img.save(FIGS / "misclassified_examples.png")


def plot_weights_or_kernels(model):
    first = next(layer for layer in model.layers if layer.optimizable)
    W = first.params["W"]
    cols, rows = 4, 4
    tile = 80
    img = Image.new("RGB", (cols * tile, rows * tile + 28), "white")
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), "Weight / kernel visualization", fill=(25, 25, 25), font=_font(15))
    if W.ndim == 2 and W.shape[0] == 784:
        count = min(16, W.shape[1])
        for i in range(count):
            arr = W[:, i].reshape(28, 28)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
            tile_img = _mnist_image(arr, scale=2).resize((tile, tile), Image.Resampling.BILINEAR)
            img.paste(tile_img, ((i % cols) * tile, 28 + (i // cols) * tile))
    elif W.ndim == 4:
        count = min(16, W.shape[0])
        for i in range(count):
            arr = W[i, 0]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
            tile_img = Image.fromarray((arr * 255).astype(np.uint8), mode="L").resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")
            img.paste(tile_img, ((i % cols) * tile, 28 + (i // cols) * tile))
    img.save(FIGS / "weight_or_kernel_visualization.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=r".\saved_models\best_model.pickle")
    args = parser.parse_args()

    FIGS.mkdir(exist_ok=True)
    model = load_model(args.model_path)
    test_imgs, test_labs = load_test_data()
    logits = model(test_imgs)
    pred = np.argmax(logits, axis=1)

    plot_confusion(confusion_matrix(pred, test_labs))
    plot_misclassified_examples(model, test_imgs, test_labs)
    plot_weights_or_kernels(model)

    print("Saved visualization figures to ./figs:")
    print("  figs/confusion_matrix.png")
    print("  figs/misclassified_examples.png")
    print("  figs/weight_or_kernel_visualization.png")


if __name__ == "__main__":
    main()
