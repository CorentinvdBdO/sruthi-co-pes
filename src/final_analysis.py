"""
This file simply load a committee to study its results.
"""
from committee import load_committee, interactive_plots, get_mean_var
from analysis import calculate_mse, plot_surface_diff, plot_surface
from launch_barrier import pash_to_dataframe, launch_barrier_type_1
import numpy as np
from nn_regression import retransform
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    features = ["epsilon", "a3"]
    # active_trained2 to 6 exist with different training sets and shapes
    model_name = "active_trained2"
    print("Loading committee ", model_name)
    committee = load_committee(model_name)
    print("Loaded committee of ", committee.models_number, " models.")
    data = pash_to_dataframe("data/pash/large_pash.dat")
    print("Loaded the test dataset. ",len(data), " elements")
    train_dataset = pd.read_csv("data/csv/"+model_name+".csv")
    print("Loaded the train dataset. ",len(train_dataset), " elements")
    print("Predicting the test set...")
    list_prediction = committee.predict(data[features])
    print("Done.")
    predicted_target, variance = get_mean_var(list_prediction)
    rmse = np.sqrt(calculate_mse(np.ravel(predicted_target), data["Barrier"]))
    predicted_target = retransform(data[features], predicted_target)
    print("rmse = ", rmse, "MeV")
    print("Dataset of size ", len(data))
    print("Train dataset of size ", len(train_dataset))

    plot_surface(predicted_target, "epsilon", "a3", "Barrier")
    plt.show()

    plot_surface_diff(data, predicted_target, "epsilon", "a3", "Barrier")
    plt.show()

    o = interactive_plots(committee, features, "Barrier", data, train_dataset, plot_train=False)

    print("Finished")