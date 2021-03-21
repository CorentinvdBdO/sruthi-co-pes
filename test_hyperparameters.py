import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nn_regression import create_datasets, build_model, normalize
from extract_pash import pash_to_dataframe

if __name__ == "__main__":
    # Get the data
    data = pash_to_dataframe("barrier/pash.dat")
    train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets(data, ["P(1)", "P(2)"], "Barrier")
    normalizer = normalize(train_features)
    layers_parameters = [1,2,2,3,4]
    neurons_parameters = [100, 200]
    fig, axs = plt.subplots(len(layers_parameters), len(neurons_parameters))

    plot_y_pos = 0

    for n_layers in layers_parameters:
        plot_x_pos = 0
        for n_neurons in neurons_parameters:
            print(str(n_layers) + "of"+ str(n_neurons))
            shape = [ n_neurons for i in range(n_layers)]
            model = build_model(normalizer, shape)
            history = model.fit(train_features, train_labels, epochs=100, verbose=0)
            axs[plot_y_pos, plot_x_pos].plot (test_labels, model.predict(test_features), '.')
            axs[plot_y_pos, plot_x_pos].set_title(str(n_layers) + "of"+ str(n_neurons))
            plot_x_pos += 1
        plot_y_pos += 1
    plt.show()
