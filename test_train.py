# An example of reading MNIST data and training models.
# This file follows the starter-code flow and saves models under ./saved_models.
import mynn as nn

import numpy as np
from struct import unpack
import gzip
import pickle
import os
import sys
import json
import shutil

try:
        import matplotlib.pyplot as plt
        from draw_tools.plot import plot
        HAS_MATPLOTLIB = True
except ImportError:
        plt = None
        plot = None
        HAS_MATPLOTLIB = False


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
os.makedirs('saved_models', exist_ok=True)
_log_file = open(r'.\logs\train_log.txt', 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, _log_file)


# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs.astype(np.float64) / 255.0
valid_imgs = valid_imgs.astype(np.float64) / 255.0


os.makedirs('figs', exist_ok=True)


def save_learning_curve(runner, name):
        # Same plotting style as the starter code, but save one figure per model
        # because this script trains several models in sequence.
        if not HAS_MATPLOTLIB:
                print(f"matplotlib is not available, skip plotting {name} learning curve")
                return
        fig, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        fig.set_tight_layout(1)
        plot(runner, axes)
        axes[0].set_title(f"{name} loss")
        axes[1].set_title(f"{name} accuracy")
        fig.savefig(os.path.join('figs', f'{name}_learning_curve.png'), dpi=180)
        plt.close(fig)


def make_experiments():
        mlp = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [0.0, 0.0])
        cnn_sgd = nn.models.Model_CNN(conv_channels=4, kernel_size=5, hidden_dim=64, weight_decay_lambda=0.0)
        cnn_momentum = nn.models.Model_CNN(conv_channels=4, kernel_size=5, hidden_dim=64, weight_decay_lambda=0.0)
        cnn_l2 = nn.models.Model_CNN(conv_channels=4, kernel_size=5, hidden_dim=64, weight_decay_lambda=1e-4)

        return [
                {
                        'name': 'mlp_sgd',
                        'model': mlp,
                        'optimizer': nn.optimizer.SGD(init_lr=0.06, model=mlp),
                        'scheduler': None,
                },
                {
                        'name': 'cnn_sgd',
                        'model': cnn_sgd,
                        'optimizer': nn.optimizer.SGD(init_lr=0.03, model=cnn_sgd),
                        'scheduler': None,
                },
                {
                        'name': 'cnn_momentum_multistep',
                        'model': cnn_momentum,
                        'optimizer': nn.optimizer.MomentGD(init_lr=0.02, model=cnn_momentum, mu=0.9),
                        'scheduler': None,
                },
                {
                        'name': 'cnn_l2',
                        'model': cnn_l2,
                        'optimizer': nn.optimizer.SGD(init_lr=0.03, model=cnn_l2),
                        'scheduler': None,
                },
        ]


experiments = make_experiments()
for exp in experiments:
        if exp['name'] == 'cnn_momentum_multistep':
                exp['scheduler'] = nn.lr_scheduler.MultiStepLR(
                        optimizer=exp['optimizer'],
                        milestones=[100, 200],
                        gamma=0.5
                )

results = []
best_score = -1.0
best_model_file = None

for exp in experiments:
        print("=" * 80)
        print(f"Start training: {exp['name']}")
        loss_fn = nn.op.MultiCrossEntropyLoss(model=exp['model'], max_classes=train_labs.max() + 1)
        runner = nn.runner.RunnerM(
                exp['model'],
                exp['optimizer'],
                nn.metric.accuracy,
                loss_fn,
                batch_size=64,
                scheduler=exp['scheduler']
        )

        # Keep the starter-code save_dir. The temporary best_model.pickle is copied
        # after each experiment so all models are preserved.
        runner.train(
                [train_imgs, train_labs],
                [valid_imgs, valid_labs],
                num_epochs=5,
                log_iters=100,
                eval_interval=100,
                save_dir=r'./saved_models'
        )

        save_learning_curve(runner, exp['name'])

        model_file = os.path.join('saved_models', f"best_model_{exp['name']}.pickle")
        shutil.copyfile(os.path.join('saved_models', 'best_model.pickle'), model_file)

        result = {
                'name': exp['name'],
                'best_valid_acc': runner.best_score,
                'model_path': model_file,
        }
        results.append(result)
        print(f"Finished {exp['name']}, best validation accuracy: {runner.best_score:.4f}")
        print(f"Saved as: {model_file}")

        if runner.best_score > best_score:
                best_score = runner.best_score
                best_model_file = model_file

if best_model_file is not None:
        shutil.copyfile(best_model_file, os.path.join('saved_models', 'best_model.pickle'))
        print("=" * 80)
        print(f"Best overall model: {best_model_file}")
        print(f"Best overall validation accuracy: {best_score:.4f}")
        print(r"Copied best overall model back to ./saved_models/best_model.pickle")

with open(r'.\logs\train_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

with open(r'.\logs\latest_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Full MNIST training summary\n")
        f.write("train: 50000\n")
        f.write("validation: 10000\n")
        f.write("epochs: 5\n")
        for result in results:
                f.write(f"{result['name']}: best_valid_acc={result['best_valid_acc']:.4f}, model={result['model_path']}\n")
        f.write(f"best_overall: {best_model_file}, best_valid_acc={best_score:.4f}\n")
