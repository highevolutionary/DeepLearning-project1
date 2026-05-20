import json
import pickle
from pathlib import Path

import numpy as np

import mynn as nn
from project_experiments import (
    prepare_data, confusion_matrix, plot_confusion, plot_examples,
    plot_weights, OUT
)


def load_model(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    model_type = state.get("model_type") if isinstance(state, dict) else "MLP"
    model = nn.models.Model_CNN() if model_type == "CNN" else nn.models.Model_MLP()
    model.load_model(path)
    return model


def main():
    results_path = OUT / "results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    _, _, test_set = prepare_data(
        train_limit=payload["train_size"],
        valid_size=payload["valid_size"],
        test_limit=payload["test_size"],
        seed=payload["seed"],
    )
    loss_fn = nn.op.MultiCrossEntropyLoss(model=None, max_classes=10)
    best_name = None
    best_acc = -1.0
    best_model = None

    for item in payload["results"]:
        model = load_model(item["model_path"])
        logits = model(test_set[0])
        acc = float(nn.metric.accuracy(logits, test_set[1]))
        loss = float(loss_fn(logits, test_set[1]))
        item["checkpoint_test_acc"] = acc
        item["checkpoint_test_loss"] = loss
        item["final_test_acc"] = acc
        item["final_test_loss"] = loss
        if acc > best_acc:
            best_acc = acc
            best_name = item["name"]
            best_model = model

    payload["best_model"] = best_name
    pred = np.argmax(best_model(test_set[0]), axis=1)
    plot_confusion(confusion_matrix(pred, test_set[1]))
    plot_examples(best_model, test_set)
    plot_weights(best_model)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print([(r["name"], r["best_valid_acc"], r["final_test_acc"], r["final_test_loss"]) for r in payload["results"]])
    print("best", payload["best_model"])


if __name__ == "__main__":
    main()
