import numpy as np
import os
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


class RunnerM():
    """
    Train, evaluate and save a model.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.history = []
        self.best_score = 0.0

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        eval_interval = kwargs.get("eval_interval", log_iters)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_score = 0
        global_step = 0

        for epoch in range(num_epochs):
            X, y = train_set
            assert X.shape[0] == y.shape[0]
            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            num_batches = int(np.ceil(X.shape[0] / self.batch_size))
            for iteration in tqdm(range(num_batches), desc=f"epoch {epoch}"):
                train_X = X[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                trn_score = self.metric(logits, train_y)
                self.train_loss.append(trn_loss)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                should_eval = (global_step % eval_interval == 0) or (iteration == num_batches - 1)
                if should_eval:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    self.history.append({
                        'epoch': epoch,
                        'iteration': iteration,
                        'step': global_step,
                        'train_loss': float(trn_loss),
                        'train_score': float(trn_score),
                        'dev_loss': float(dev_loss),
                        'dev_score': float(dev_score),
                        'lr': float(self.optimizer.init_lr)
                    })
                    if iteration % log_iters == 0 or iteration == num_batches - 1:
                        print(f"epoch: {epoch}, iteration: {iteration}, step: {global_step}")
                        print(f"[Train] loss: {trn_loss:.6f}, score: {trn_score:.6f}")
                        print(f"[Dev] loss: {dev_loss:.6f}, score: {dev_score:.6f}")
                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score
                global_step += 1
        self.best_score = best_score

    def evaluate(self, data_set, batch_size=None):
        X, y = data_set
        batch_size = batch_size or self.batch_size
        losses = []
        scores = []
        weights = []
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            batch_X = X[start:end]
            batch_y = y[start:end]
            logits = self.model(batch_X)
            losses.append(self.loss_fn(logits, batch_y))
            scores.append(self.metric(logits, batch_y))
            weights.append(batch_X.shape[0])
        weights = np.asarray(weights, dtype=np.float64)
        return float(np.average(scores, weights=weights)), float(np.average(losses, weights=weights))

    def save_model(self, save_path):
        self.model.save_model(save_path)

