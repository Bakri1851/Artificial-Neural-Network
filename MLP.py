import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MLP:
    def __init__(self, weight_range, hidden_node_amount):
        self.input_values = []
        self.expected_output_value = 0
        # Will store predictors and predictand

        self.input_to_hidden = np.random.uniform(-weight_range, weight_range, size=(9, hidden_node_amount))
        self.hidden_to_out = np.random.uniform(-weight_range, weight_range, size=(hidden_node_amount + 1,))
        # Randomise weights and biases

        self.hidden_node_amount = hidden_node_amount
        # Store amount of hidden nodes

        self.activations = []
        self.output_activation = 0
        # Will store activations which we will use on forward pass

        self.delta_values = []
        # Will store delta values which we will use on backward pass

        self.temp = 0
        self.change = 0
        self.temp2 = 0
        self.change2 = 0
        # Store previous and change to be used to implement momentum

    def plug_row_in(self, input_values, expected_output_value):
        self.input_values = np.insert(input_values, 0, 1)
        self.expected_output_value = expected_output_value
        # Will store the current predictors and predictand

    def forward_pass(self):
        sums = np.matmul(self.input_values, self.input_to_hidden)
        self.activations = np.insert(sigmoid(sums), 0, 1)
        output_sum = np.matmul(self.activations, self.hidden_to_out)
        self.output_activation = sigmoid(output_sum)
        # Uses matrix and vector multiplication to allow for forward pass

    def backward_pass(self):
        derivative = self.output_activation * (1 - self.output_activation)
        self.output_delta_value = (self.expected_output_value - self.output_activation) * derivative
        self.delta_values = []
        for i in range(1, len(self.activations)):
            derivative = self.activations[i] * (1 - self.activations[i])
            self.delta_values.append(self.hidden_to_out[i] * self.output_delta_value * derivative)
        # Run backward pass by using delta values and applying equations using activations

    def update_weights_and_biases(self, x, r):
        p = 0.01
        q = 0.1
        learning_rate = p + (q - p) * (1 - (1 / (1 + np.exp(10 - (20 * x) / r))))
        # Annealing

        alpha = 0.9
        self.temp = self.input_to_hidden
        self.input_to_hidden = self.input_to_hidden + learning_rate * np.outer(self.input_values,
                                                                               self.delta_values) + alpha * self.change
        self.change = self.input_to_hidden - self.temp
        self.temp2 = self.hidden_to_out
        self.hidden_to_out = self.hidden_to_out + learning_rate * self.activations * self.output_delta_value + alpha * self.change2
        self.change2 = self.hidden_to_out - self.temp2
        # Momentum
        # Will Change the hidden to output weights and output bias
