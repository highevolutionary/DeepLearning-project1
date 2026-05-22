# plot the score and loss
import matplotlib.pyplot as plt

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    train_epochs = [i for i in range(len(runner.train_scores))]
    if getattr(runner, 'history', None) and len(runner.history) == len(runner.dev_scores):
        dev_epochs = [item.get('step', i) for i, item in enumerate(runner.history)]
    else:
        dev_epochs = [i for i in range(len(runner.dev_scores))]
    axes[0].plot(train_epochs, runner.train_loss, color=train_color, label="Train loss")
    axes[0].plot(dev_epochs, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')

    axes[1].plot(train_epochs, runner.train_scores, color=train_color, label="Train accuracy")
    axes[1].plot(dev_epochs, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')
