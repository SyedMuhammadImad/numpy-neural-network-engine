"""Retained project engine with assisted completion repairs; see ATTRIBUTION.md."""
import numpy as np
SEED=42

class NeuralNetwork:

    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    @staticmethod
    def mean_squared_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def accuracy(y_pred, y_true):
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    def Leaky_ReLU(self, x):
        return np.where(x > 0, x, x * self.leaky_slope)

    def Tanh(self, x):
        return np.tanh(x)

    def __init__(self, nodes_per_layer: list[int], mode: str='classification', optimizer: str='fullbatch', activation: str='sigmoid', batch_size: int=64):
        """Creates a Feed-Forward Neural Network.
        "nodes_per_layer" is a list containing the number of nodes in each layer (including the input layer) e.g. [2, 10, 10, 2] (2 features, 2 hidden layers with 10 neurons each, 2 classes)"
        """
        if any((type(n) is not int for n in nodes_per_layer)):
            raise ValueError('Layer sizes must be integers')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Invalid batch size')
        if len(nodes_per_layer) < 2:
            raise ValueError('Network must have at least 2 layers (input and output).')
        if not (np.array(nodes_per_layer) > 0).all():
            raise ValueError('The number of nodes in all layers must be positive.')
        if mode.lower() not in ('classification', 'regression'):
            raise ValueError('Invalid mode')
        if optimizer.lower() not in ('sgd', 'minibatch', 'fullbatch'):
            raise ValueError('Invalid optimizer')
        if activation.lower() not in ('sigmoid', 'relu', 'leaky_relu', 'tanh'):
            raise ValueError('Invalid activation')
        self.num_layers = np.size(nodes_per_layer)
        self.mode = mode.lower()
        self.nodes_per_layer = nodes_per_layer
        self.input_shape = nodes_per_layer[0]
        self.output_shape = nodes_per_layer[-1]
        self.optimizer = optimizer.lower()
        self.activation_strat = activation.lower()
        self.batch_size = batch_size
        if activation.lower() == 'leaky_relu':
            self.leaky_slope = 0.05
        self.weights_ = []
        self.biases_ = []
        self.__init_weights(nodes_per_layer)

    def __init_weights(self, nodes_per_layer):
        """Initializes all weights based on a standard normal distribution and all biases to 0."""
        rng = np.random.default_rng(SEED)
        for i, _ in enumerate(nodes_per_layer):
            if i == 0:
                continue
            weight_matrix = rng.normal(size=(nodes_per_layer[i - 1], nodes_per_layer[i])) * np.sqrt((2.0 if self.activation_strat in ('relu', 'leaky_relu') else 1.0) / nodes_per_layer[i - 1])
            self.weights_.append(weight_matrix)
            bias_vector = np.zeros(shape=(nodes_per_layer[i],))
            self.biases_.append(bias_vector)

    def fit(self, Xs, Ys, X_val, Y_val, epochs, lr=0.001):
        Xs = self._features(Xs)
        X_val = self._features(X_val)
        Ys = self._targets(Ys, len(Xs))
        Y_val = self._targets(Y_val, len(X_val))
        if type(epochs) is not int or epochs < 1 or (not np.isfinite(lr)) or (lr <= 0):
            raise ValueError('Invalid training settings')
        self.history = []
        self.val_history = []
        self._run_optimizer(Xs, Ys, X_val, Y_val, epochs, lr)
        return (self.history, self.val_history)

    def _run_optimizer(self, Xs, Ys, X_val, Y_val, epochs, lr):
        rng = np.random.default_rng(SEED)
        best_loss = np.inf
        best_weights = None
        best_biases = None
        size = len(Xs) if self.optimizer == 'fullbatch' else 1 if self.optimizer == 'sgd' else self.batch_size
        for epoch in range(epochs):
            indices = np.arange(len(Xs)) if self.optimizer == 'fullbatch' else rng.permutation(len(Xs))
            for start in range(0, len(Xs), size):
                batch = indices[start:start + size]
                x, y = (Xs[batch], Ys[batch])
                activations = self.forward_pass(x)
                deltas = self.backward_pass(y, activations)
                self.weight_update(deltas, [x] + activations[:-1], lr)
            loss_fn = self.cross_entropy_loss if self.mode == 'classification' else self.mean_squared_loss
            self.history.append(float(loss_fn(self.predict(Xs), Ys)))
            loss = float(loss_fn(self.predict(X_val), Y_val))
            self.val_history.append(loss)
            if not np.isfinite([self.history[-1], loss]).all():
                raise ValueError('Training diverged')
            if loss < best_loss:
                best_loss = loss
                self.selected_epoch = epoch + 1
                best_weights = [w.copy() for w in self.weights_]
                best_biases = [b.copy() for b in self.biases_]
        self.weights_, self.biases_ = (best_weights, best_biases)

    def forward_pass(self, input_data):
        input_data = self._features(input_data)
        activations = []
        curr_input = input_data
        for i in range(self.num_layers - 1):
            z = np.dot(curr_input, self.weights_[i]) + self.biases_[i]
            if i == self.num_layers - 2:
                if self.mode == 'classification':
                    a = self.softmax(z)
                else:
                    a = z
            elif self.activation_strat == 'sigmoid':
                a = self.sigmoid(z)
            elif self.activation_strat == 'relu':
                a = self.ReLU(z)
            elif self.activation_strat == 'leaky_relu':
                a = self.Leaky_ReLU(z)
            elif self.activation_strat == 'tanh':
                a = self.Tanh(z)
            else:
                a = z
            activations.append(a)
            curr_input = a
        return activations

    def backward_pass(self, targets, layer_activations):
        deltas = []
        num_samples = targets.shape[0]
        output = layer_activations[-1]
        if self.mode == 'classification':
            delta = (output - targets) / num_samples
        else:
            delta = 2 * (output - targets) / output.size
        deltas.append(delta)
        for i in range(self.num_layers - 2, 0, -1):
            delta_prev_layer = np.dot(deltas[-1], self.weights_[i].T)
            a = layer_activations[i - 1]
            if self.activation_strat == 'sigmoid':
                derivative = a * (1 - a)
            elif self.activation_strat == 'relu':
                derivative = (a > 0).astype(float)
            elif self.activation_strat == 'leaky_relu':
                derivative = np.where(a > 0, 1.0, self.leaky_slope)
            elif self.activation_strat == 'tanh':
                derivative = 1 - a ** 2
            else:
                derivative = 1.0
            delta = delta_prev_layer * derivative
            deltas.append(delta)
        return deltas[::-1]

    def weight_update(self, deltas, layer_inputs, lr):
        for i in range(len(self.weights_)):
            w_grad = np.dot(layer_inputs[i].T, deltas[i])
            b_grad = np.sum(deltas[i], axis=0)
            self.weights_[i] -= lr * w_grad
            self.biases_[i] -= lr * b_grad

    def predict(self, Xs):
        """Returns the model predictions (output of the last layer) for the given "Xs"."""
        return self.forward_pass(Xs)[-1]

    def evaluate(self, Xs, Ys):
        preds = self.predict(Xs)
        targets = self._targets(Ys, len(preds))
        if self.mode == 'regression':
            return self.mean_squared_loss(preds, targets)
        return (self.cross_entropy_loss(preds, targets), self.accuracy(preds, targets))

    def _features(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not len(x) or x.shape[1] != self.input_shape or (not np.isfinite(x).all()):
            raise ValueError('Expected nonempty finite matrix with matching input features')
        return x

    def _targets(self, y, rows):
        y = np.asarray(y)
        if self.mode == 'classification' and y.ndim == 1:
            if y.dtype.kind not in 'iu' or len(y) != rows or (y < 0).any() or (y >= self.output_shape).any():
                raise ValueError('Invalid integer classification targets')
            y = np.eye(self.output_shape)[y]
        elif self.mode == 'regression' and y.ndim == 1 and (self.output_shape == 1):
            y = y[:, None]
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (rows, self.output_shape) or not np.isfinite(y).all():
            raise ValueError('Invalid target shape or values')
        if self.mode == 'classification' and ((y < 0).any() or not np.allclose(y.sum(axis=1), 1)):
            raise ValueError('Classification targets must be distributions summing to one')
        return y
