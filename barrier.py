import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import tensorflow as tf
import extract_pash
from launch_barrier import launch_barrier, input_template, change_file_name
from extract_pash import pash_to_dataframe, plot_surface
from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform, plot_surface_diff


#input_template("step3")
#launch_barrier()
#change_file_name("pash.dat", "pash_step3new.dat")
dataset = pash_to_dataframe("barrier/pash_step3new.dat")
features = ["P(1)", "P(2)"]
train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets(dataset, features, "Barrier")
normalizer = normalize(train_features)
model = build_model(normalizer, [100, 100, 100])
epoch_no = 2000
history = model.fit(train_features, train_labels, epochs=epoch_no).history
predicted_labels = model.predict(dataset[features])
predicted_dataset = retransform(dataset[features], predicted_labels)
fig = plt.figure()
ax = fig.gca(projection='3d')
plot_surface(dataset, "P(1)", "P(2)", "Barrier", ax, alpha=0.5)
plot_surface(predicted_dataset,"P(1)", "P(2)", "Barrier", ax,alpha=0.5)
plot_surface_diff(dataset,predicted_dataset,"P(1)", "P(2)", "Barrier", ax)
plt.show()
learning_curve(history)
plt.show()



