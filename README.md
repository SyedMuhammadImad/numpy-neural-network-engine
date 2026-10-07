# Manual neural-network classification and regression

Completed general-purpose experimental feed-forward network implementation with explicit NumPy forward propagation, backpropagation and weight updates. It supports classification/regression, sigmoid/ReLU/leaky-ReLU/tanh and stochastic, minibatch and full-batch gradient descent. It is a author algorithm experiment, not an original research result or production ML service. See `ATTRIBUTION.md`.

## Run

Use Python 3.12, install `requirements.txt`, then:

```text
python -m pytest -q
python experiment.py --data-dir /path/to/local-fashion-mnist --output-dir /path/to/local-results --download
```

`--download` explicitly obtains Fashion-MNIST through torchvision's checksum-verified loader; omit it when data is already local. The experiment also generates the documented synthetic moons and Eggholder tasks. Original datasets, Fashion-MNIST images/pixels, checkpoints, plots, credentials and media are not committed. Keep outputs outside the publication tree and only load your own trusted checkpoints.

For programmatic use, construct `network.NeuralNetwork(nodes_per_layer, mode, optimizer, activation, batch_size)` and call `fit(train_x, train_y, validation_x, validation_y, epochs, lr)`. It returns training/validation histories and restores the lowest-validation-loss checkpoint. Classification targets may be integer class indices 0..C−1 or valid target distributions; regression supports one or several output dimensions. `predict` returns class probabilities or regression values. `evaluate` returns cross-entropy/accuracy or MSE. Input dimensions, targets and training settings are checked explicitly.

For stored experiments, `experiment.predict(local_results_directory, model_name, raw_feature_matrix)` reloads the local checkpoint, applies its fitted feature scale and restores original target units for regression. This repairs the original notebook's use of raw inputs against a normalized regression model. Scale statistics are fitted on training data only. Stable sigmoid and variance-scaled initialization avoid overflow; multi-output MSE backprop now divides by every target element, matching the actual mean loss.

## Fresh complete experiments

All experiments use seed 42, validation checkpoint selection and disjoint held-out test data. The model configurations and complete histories are in `metrics.json`. No test accuracy or R² selected an architecture, learning rate or checkpoint.

The five original optimizer/activation configurations run on a synthetic 1,800-sample moons dataset (noise 0.2), with 1,080 train / 360 validation / 360 test samples:

- fullbatch_sigmoid: 89.44% test accuracy
- sgd_tanh: 97.78% test accuracy
- minibatch_relu: 97.50% test accuracy
- minibatch_leaky_relu: 96.94% test accuracy
- minibatch_relu_high_lr: 96.94% test accuracy

The three regression configurations run on a synthetic noisy Eggholder function, also 1,080/360/360. Inputs and targets use training-only scaling; reported MSE and R² are in original target units:

- eggholder_tanh: 0.552 held-out R²
- eggholder_relu: 0.141 held-out R²
- eggholder_sigmoid: 0.005 held-out R²

These regression results range from moderate to weak. The sigmoid configuration barely improves over the training-mean baseline; the implementation is complete, but that does not make every model a good approximation. Baseline MSE and actual measured errors are included.

The manual 784→20→20→10 ReLU Fashion-MNIST network trained for 30 epochs on 18,000 examples, selected its checkpoint with 2,000 separate validation examples, and evaluated all 10,000 official test examples: 84.79% accuracy. This explicit NumPy/backprop example differs from the separate framework MLP/CNN comparison. The original notebook's use of the official test set as validation was repaired.

All saved experiment checkpoints were reloaded and reproduced held-out outputs exactly. These are one-seed experimental measurements; synthetic-task results do not establish performance on real applications or broad deployment reliability.

## Verification

Sixteen independent tests check every parameter gradient for all four activations in classification and multi-output regression, all three optimizers, best-validation checkpoint restoration, integer/one-hot parity, learned linear regression, scale-correct saved-model inference, split separation, finite extremes and rejected settings. The notebook follows setup → mathematical check → experiment → results and executes every code cell without embedding media. `ATTRIBUTION.md` retains supplied example context. See `VERIFICATION.json`.
