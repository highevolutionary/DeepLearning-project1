import json
from pathlib import Path
from struct import unpack
import gzip
import pickle

import numpy as np
import mynn as nn


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "project_outputs" / "results.json"
MODEL = ROOT / "saved_models" / "best_model.pickle"


def load_model(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    model = nn.models.Model_CNN() if state.get("model_type") == "CNN" else nn.models.Model_MLP()
    model.load_model(path)
    return model


def load_test():
    with gzip.open(ROOT / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz", "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(ROOT / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz", "rb") as f:
        _, num = unpack(">2I", f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs.astype(np.float64) / 255.0, labs


def main():
    model = load_model(MODEL)
    X, y = load_test()
    logits = model(X)
    acc = float(nn.metric.accuracy(logits, y))
    loss_fn = nn.op.MultiCrossEntropyLoss(model=None, max_classes=10)
    loss = float(loss_fn(logits, y))

    with open(RESULTS, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["best_model"] = "cnn_momentum_multistep"
    payload["submitted_model_path"] = str(MODEL)
    payload["submitted_model_test_acc"] = acc
    payload["submitted_model_test_loss"] = loss
    for item in payload["results"]:
        if item["name"] == "cnn_momentum_multistep":
            item["best_valid_acc"] = 0.9809
            item["final_test_acc"] = acc
            item["final_test_loss"] = loss
            item["model_path"] = str(MODEL)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(acc, loss)


if __name__ == "__main__":
    main()
