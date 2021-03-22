import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from launch_barrier import launch_barrier, input_template
from extract_pash import pash_to_dataframe
from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
from hyperparameters import calculate_rmse, rmse_test, hyperparameter_analysis

dataset = pash_to_dataframe("barrier/pash.dat")
features = ["epsilon", "a3"]
train_dataset, test_dataset, \
        train_features, train_labels, \
        test_features, test_labels \
        = create_datasets(dataset, features, "Barrier", frac=0.1)
normalizer = normalize(train_features)
model = build_model(normalizer, [100, 100, 100])
epoch_no = 1000
history = model.fit(train_features, train_labels, epochs=epoch_no).history
predicted_labels = model.predict(dataset[features])
expected_labels = dataset.pop('Barrier')
rmse = calculate_rmse(predicted_labels, expected_labels)
print(rmse)
