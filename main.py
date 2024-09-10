import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLP import MLP
from scipy.stats import linregress

if __name__ == '__main__':
    filename = "FEHDataStudent.xlsx"
    df = pd.read_excel(filename, sheet_name="Train")
    input_values = df.iloc[:, :8].values
    output_value = df.iloc[:, 8].values

    # Take in the predictors and predictands from the Train set and store in input_values and output_value respectively

    df = pd.read_excel(filename, sheet_name="Val")
    input_values_v = df.iloc[:, :8].values
    output_value_v = df.iloc[:, 8].values

    # Take in the predictors and predictands from the Val set and store in input_values_v and output_value_v respectively

    df = pd.read_excel(filename, sheet_name="Test")
    input_values_t = df.iloc[:, :8].values
    output_value_t = df.iloc[:, 8].values

    # Take in the predictors and predictands from the Test set and store in input_values_t and output_value_t respectively

    hidden_nodes = 16
    # Adjust nodes to experiment to see optimal amount. Choose betweeen 4-16
    epochs = 100
    # 2500 epochs is enough to clearly see how the RMSE is affected as you get more epochs
    # After this amount we get diminishing roles
    weight_range = 0.25
    # Since we have 8 inputs we use a weight range of [-0.25,0.25]
    # 2/n where n = 8

    my_MLP = MLP(weight_range, hidden_nodes)
    # Create an instance of the MLP class
    # This is the MLP which we will train

    rmse_values = []
    rmse_values_v = []
    # Create lists that store the RMSE values on each epoch. These values will then be used to plot the graph
    for i in range(epochs):
        temp = 0
        for j in range(353):
            my_MLP.plug_row_in(input_values[j], output_value[j])
            my_MLP.forward_pass()
            my_MLP.backward_pass()
            my_MLP.update_weights_and_biases(i, epochs)
            temp += (my_MLP.expected_output_value - my_MLP.output_activation) ** 2
        mse = temp / 353
        rmse_values.append(mse ** 0.5)

        # On the training data, we will forward pass, backward pass then update weights
        # We will calculate an RMSE from this which we store in rmse_values

        temp = 0
        for j in range(118):
            my_MLP.plug_row_in(input_values_v[j], output_value_v[j])
            my_MLP.forward_pass()
            temp += (my_MLP.expected_output_value - my_MLP.output_activation) ** 2

        mse_v = temp / 118
        rmse_values_v.append(mse_v ** 0.5)
        # On the training data, we will forward pass
        # We will calculate an RMSE from this which we store in rmse_values
        print(f"Epoch: {i}, Train RMSE: {mse}, Val RMSE: {mse_v}")

    test_expected = []
    test_calculated = []
    for i in range(118):
        my_MLP.plug_row_in(input_values_t[i], output_value_t[i])
        my_MLP.forward_pass()
        test_expected.append(my_MLP.expected_output_value)
        test_calculated.append(my_MLP.output_activation)


    plt.plot(range(1, epochs + 1), rmse_values, label="Train")
    plt.plot(range(1, epochs + 1), rmse_values_v, label="Val")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error vs. Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()


    test_calculated = np.array(test_calculated)
    test_expected = np.array(test_expected)
    slope, intercept, r_value, p_value, std_err = linregress(test_calculated, test_expected)

    # Create scatter plot
    plt.scatter(test_calculated, test_expected)  # Use scatter() for scatter plot

    # Plot the line of best fit
    plt.plot(test_calculated, slope * test_calculated + intercept, color='red')
    plt.xlabel('Calculated Value')
    plt.ylabel('Expected Value')
    plt.title('Expected vs. Calculated')
    plt.grid(True)
    plt.show()

