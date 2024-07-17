import numpy as np
import matplotlib.pyplot as plt

# Defining params
WEIGHTS = np.array([np.random.rand(), np.random.rand()])
BIAS = np.random.rand()
LR = 0.1

input_vectrs = np.array([[1.66, 1.56], [2, 1.5]])
targets = np.array([1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_pred(input):
    layer_1 = np.dot(input, WEIGHTS) + BIAS
    layer_2 = sigmoid(layer_1)
    return layer_2


def output(pred):
    if pred > 0.5:
        return 1
    return 0


print(f'Input 1: {output(make_pred(input_vectrs[0]))}')
print(f'Input 2: {output(make_pred(input_vectrs[1]))}')  # <-- Here we can see the output is 1, which is incorrect


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def compute_gradients(input, target):
    layer_1 = np.dot(input, WEIGHTS) + BIAS
    layer_2 = sigmoid(layer_1)
    pred = layer_2

    # Prediction error between the target & predicted output
    pred_error = 2 * (pred - target)

    # Derivative of Sigmoid layer
    pred_layer1 = sigmoid_derivative(layer_1)

    # Derivative of Bias
    layer1_bias = 1

    # Derivative of weights
    layer1_weights = (0 * WEIGHTS) + (1 * input)

    # Update the error values
    error_bias = (pred_error * pred_layer1 * layer1_bias)
    error_weights = (pred_error * layer1_weights)

    return error_bias, error_weights


def update_parameters(error_bias, error_weights, bias, weights):
    bias = bias - (error_bias * LR)
    weights = weights - (error_weights * LR)


# Compute the error_bias and error_weights
error_bias, error_weights = compute_gradients(input_vectrs[1], targets[1])
BIAS = BIAS - (error_bias * LR)
WEIGHTS = WEIGHTS - (error_weights * LR)

print(f"Input 2: {output(make_pred(input_vectrs[1]))}")
print(f"Input 1: {output(make_pred(input_vectrs[0]))}")


def train(inputs, targets, iterations, bias, weights):
    cumulative_errors = []

    # iteration in the given range
    for current_iteration in range(iterations):
        # pick a data instance for at random.
        random_index = np.random.randint(len(inputs))
        input_vector = inputs[random_index]
        target = inputs[random_index]

        # compute the gradients to update the weights
        error_bias, error_weights = compute_gradients(input_vector, target)
        update_parameters(error_bias, error_weights, bias, weights)

        # Measure the cumulative error for all instances
        if current_iteration % 100 == 0:
            cumulative_error = 0
            for index in range(len(inputs)):
                data_point = inputs[index]
                target = targets[index]

                prediction = make_pred(data_point)
                error = np.square(prediction - target)

                cumulative_error += error
                cumulative_errors.append(cumulative_error)

    return cumulative_errors


training_error = train(input_vectrs, targets, 10000, BIAS, WEIGHTS)


def plot_error(error):
    plt.plot(error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.show()


input_vectors = np.array([
    [3, 1.5],
    [2, 1],
    [4, 1.5],
    [3, 4],
    [3.5, 0.5],
    [2, 0.5],
    [5.5, 1],
    [1, 1]
])

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
weights = np.array([[np.random.rand()], [np.random.rand()]])
bias = np.random.rand()
training_error = train(input_vectors, targets, 100, weights, bias)
plot_error(training_error)