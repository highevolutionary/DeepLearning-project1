from .op import *
import pickle


class Model_MLP(Layer):
    """
    A model with linear layers.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                weight_decay = lambda_list is not None and lambda_list[i] is not None and lambda_list[i] > 0
                layer = Linear(
                    in_dim=size_list[i],
                    out_dim=size_list[i + 1],
                    weight_decay=weight_decay,
                    weight_decay_lambda=(lambda_list[i] if lambda_list is not None else 0.0)
                )
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        raise NotImplementedError
                    elif act_func == 'ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise ValueError(f"unknown activation: {act_func}")

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, \
            'Model has not initialized yet. Use model.load_model to load a model or create a new model.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        if isinstance(param_list, dict):
            self.size_list = param_list['size_list']
            self.act_func = param_list['act_func']
            params = param_list['params']
        else:
            self.size_list = param_list[0]
            self.act_func = param_list[1]
            params = param_list[2:]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            item = params[i]
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = item['W']
            layer.b = item['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = item.get('weight_decay', False)
            layer.weight_decay_lambda = item.get('lambda', item.get('weight_decay_lambda', 0.0))
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplementedError
                elif self.act_func == 'ReLU':
                    self.layers.append(ReLU())

    def save_model(self, save_path):
        param_list = {'model_type': 'MLP', 'size_list': self.size_list, 'act_func': self.act_func, 'params': []}
        for layer in self.layers:
            if layer.optimizable:
                param_list['params'].append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    Lightweight CNN for MNIST: conv2D -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
    """
    def __init__(self, in_channels=1, num_classes=10, conv_channels=4, kernel_size=5,
                 hidden_dim=64, weight_decay_lambda=0.0):
        super().__init__()
        self.config = {
            'in_channels': in_channels,
            'num_classes': num_classes,
            'conv_channels': conv_channels,
            'kernel_size': kernel_size,
            'hidden_dim': hidden_dim,
            'weight_decay_lambda': weight_decay_lambda
        }
        self.input_shape = (in_channels, 28, 28)
        self.layers = []
        if in_channels is not None:
            self._build_layers()

    def _build_layers(self):
        c = self.config['conv_channels']
        k = self.config['kernel_size']
        hidden = self.config['hidden_dim']
        num_classes = self.config['num_classes']
        wd = self.config.get('weight_decay_lambda', 0.0)
        use_wd = wd is not None and wd > 0
        conv_out_h = 28 - k + 1
        conv_out_w = 28 - k + 1
        flat_dim = c * conv_out_h * conv_out_w
        self.layers = [
            conv2D(1, c, k, stride=1, padding=0, weight_decay=use_wd, weight_decay_lambda=wd),
            ReLU(),
            Flatten(),
            Linear(flat_dim, hidden, weight_decay=use_wd, weight_decay_lambda=wd),
            ReLU(),
            Linear(hidden, num_classes, weight_decay=use_wd, weight_decay_lambda=wd),
        ]

    def __call__(self, X):
        return self.forward(X)

    def _prepare_input(self, X):
        if X.ndim == 2:
            return X.reshape(X.shape[0], 1, 28, 28)
        if X.ndim == 4:
            return X
        raise ValueError("Model_CNN expects flattened MNIST [N, 784] or image tensor [N, 1, 28, 28]")

    def forward(self, X):
        outputs = self._prepare_input(X)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            state = pickle.load(f)
        if not isinstance(state, dict) or state.get('model_type') != 'CNN':
            raise ValueError("the checkpoint is not a CNN model")
        self.config = state['config']
        self.input_shape = tuple(state.get('input_shape', (1, 28, 28)))
        self._build_layers()
        params = state['params']
        idx = 0
        for layer in self.layers:
            if layer.optimizable:
                item = params[idx]
                layer.W = item['W']
                layer.b = item['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = item.get('weight_decay', False)
                layer.weight_decay_lambda = item.get('lambda', item.get('weight_decay_lambda', 0.0))
                idx += 1

    def save_model(self, save_path):
        state = {'model_type': 'CNN', 'config': self.config, 'input_shape': self.input_shape, 'params': []}
        for layer in self.layers:
            if layer.optimizable:
                state['params'].append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
