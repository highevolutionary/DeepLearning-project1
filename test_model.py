import argparse
from pathlib import Path
from struct import unpack
import gzip
import pickle
import os
import sys

import mynn as nn
import numpy as np


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


os.makedirs('logs', exist_ok=True)
_log_file = open(r'.\logs\test_log.txt', 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, _log_file)


MODEL_FILES = [
    ("MLP + SGD", r".\saved_models\best_model_mlp_sgd.pickle"),
    ("CNN + SGD", r".\saved_models\best_model_cnn_sgd.pickle"),
    ("CNN + Momentum + MultiStepLR", r".\saved_models\best_model_cnn_momentum_multistep.pickle"),
    ("CNN + L2", r".\saved_models\best_model_cnn_l2.pickle"),
    ("Best overall", r".\saved_models\best_model.pickle"),
]


def load_model(path):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    model_type = state.get('model_type') if isinstance(state, dict) else 'MLP'
    model = nn.models.Model_CNN() if model_type == 'CNN' else nn.models.Model_MLP()
    model.load_model(path)
    return model


def load_test_data():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs.astype(np.float64) / 255.0
    return test_imgs, test_labs


def evaluate_one(name, path, test_imgs, test_labs):
    model = load_model(path)
    logits = model(test_imgs)
    acc = nn.metric.accuracy(logits, test_labs)
    print(f"{name}: {acc:.4f} ({path})")
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None)
    args = parser.parse_args()

    test_imgs, test_labs = load_test_data()

    if args.model_path is not None:
        evaluate_one("Specified model", args.model_path, test_imgs, test_labs)
    else:
        for name, path in MODEL_FILES:
            if Path(path).exists():
                evaluate_one(name, path, test_imgs, test_labs)
            else:
                print(f"{name}: missing ({path})")
