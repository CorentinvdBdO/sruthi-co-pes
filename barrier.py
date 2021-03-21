import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import tensorflow as tf
import extract_pash
from launch_barrier import launch_barrier, input_template, change_file_name
from extract_pash import pash_to_dataframe, plot_surface
from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform


input_template("step3")
#launch_barrier()
#change_file_name("pash.dat", "pash_step3new.dat")
pash_data = pash_to_dataframe("barrier/pash_step3new.dat")
dataset = pash_to_dataframe("barrier/pash_step3new.dat")
train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets(dataset, ["P(1)", "P(2)"], "Barrier")
normalizer = normalize(train_features)
model = build_model(normalizer, [100, 100, 100])
epoch_no = 1000
history = model.fit(train_features, train_labels, epochs=epoch_no).history
predicted_dataset = retransform(model, ["P(1)", "P(2)"], dataset)
fig = plt.figure()
ax = fig.gca(projection='3d')
plot_surface(pash_data, "P(1)", "P(2)", "Barrier", ax)
plot_surface(predicted_dataset,"P(1)", "P(2)", "Barrier", ax)
plt.show()
learning_curve(history)
plt.show()



