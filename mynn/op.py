from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Linear(Layer):
    """
    Fully connected layer.
    """
    def __init__(self, in_dim, out_dim, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is None:
            scale = np.sqrt(2.0 / in_dim)
            self.W = np.random.randn(in_dim, out_dim) * scale
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim]
        output: [batch_size, in_dim]
        """
        assert self.input is not None, "forward must be called before backward"
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params['W'].T

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class conv2D(Layer):
    """
    A 2D convolution layer for NCHW tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        k = self.kernel_size
        if initialize_method is None:
            scale = np.sqrt(2.0 / (in_channels * k * k))
            self.W = np.random.randn(out_channels, in_channels, k, k) * scale
            self.b = np.zeros((1, out_channels, 1, 1))
        else:
            self.W = initialize_method(size=(out_channels, in_channels, k, k))
            self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.input_padded = None
        self.cols = None
        self.input_shape = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def _im2col(self, X):
        N, C, H, W = X.shape
        k = self.kernel_size
        out_h = (H - k) // self.stride + 1
        out_w = (W - k) // self.stride + 1
        cols = np.empty((N, out_h, out_w, C, k, k), dtype=X.dtype)
        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                cols[:, i, j, :, :, :] = X[:, :, hs:hs + k, ws:ws + k]
        return cols.reshape(N * out_h * out_w, C * k * k), out_h, out_w

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        output: [batch, out_channels, new_H, new_W]
        """
        if X.ndim != 4:
            raise ValueError("conv2D expects input with shape [N, C, H, W]")
        self.input = X
        self.input_shape = X.shape
        if self.padding > 0:
            self.input_padded = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.input_padded = X
        self.cols, out_h, out_w = self._im2col(self.input_padded)
        W_col = self.params['W'].reshape(self.out_channels, -1)
        out = self.cols @ W_col.T
        out = out.reshape(X.shape[0], out_h, out_w, self.out_channels)
        out = out.transpose(0, 3, 1, 2) + self.params['b']
        return out

    def backward(self, grads):
        """
        grads: [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None and self.cols is not None, "forward must be called before backward"
        N, _, out_h, out_w = grads.shape
        k = self.kernel_size
        grad_out = grads.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, self.out_channels)

        self.grads['W'] = (grad_out.T @ self.cols).reshape(self.params['W'].shape)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        W_col = self.params['W'].reshape(self.out_channels, -1)
        dcols = grad_out @ W_col
        dcols = dcols.reshape(N, out_h, out_w, self.in_channels, k, k)
        dX_padded = np.zeros_like(self.input_padded)
        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                dX_padded[:, :, hs:hs + k, ws:ws + k] += dcols[:, i, j, :, :, :]

        if self.padding > 0:
            return dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_padded

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.where(X < 0, 0, X)

    def backward(self, grads):
        assert self.input.shape == grads.shape
        return np.where(self.input < 0, 0, grads)


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class MultiCrossEntropyLoss(Layer):
    """
    Multi-class cross entropy loss. The default form includes softmax.
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels: [batch_size,]
        """
        labels = labels.astype(np.int64)
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]
        if batch_size == 0:
            raise ValueError("empty batch is not supported")

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        eps = 1e-12
        correct_probs = self.probs[np.arange(batch_size), labels]
        return -np.mean(np.log(correct_probs + eps))

    def backward(self):
        batch_size = self.predicts.shape[0]
        if self.has_softmax:
            self.grads = self.probs.copy()
            self.grads[np.arange(batch_size), self.labels] -= 1
            self.grads /= batch_size
        else:
            one_hot = np.zeros_like(self.predicts)
            one_hot[np.arange(batch_size), self.labels] = 1
            self.grads = -one_hot / (self.predicts + 1e-12) / batch_size

        if self.model is not None:
            self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2 regularization is implemented through weight_decay in trainable layers.
    """
    pass


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
