import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform, create_empty_dataset
from extract_pash import pash_to_dataframe, plot_surface, plot_heatmap, plot_contour, plot_points
from committee import Committee, get_mean_var, multiple_plots
from hyperparameters import calculate_mse
from launch_barrier_type1 import launch_barrier_type_1

'''Inputs'''
goal_variance = 1                 # float        Goal for the maximum variance
goal_rmse = 0.2                   # float        Goal for the root mean squared error
features = ["epsilon", "a3"]      # [str]        Key argument of the features
target = "Barrier"                # str          Key argument of the target
frac = 120                        # int or float fraction of the large pash turning into initial training data
n_models = 5                      # int          Number of models in the committee
model_shape = [150, 150, 150, 150, 150, 150]  #  Shape of the neural networks
epochs = 200                      # int          Number of epochs per loop
batch_size = 30                   # int          Batch size during the fit
split_train = True                # Bool         Split the train set between the models
bootstrap = 0.8                   # float        Overwrites split_train : bootstraps with this fraction
top_variance_percentage = 0.2     # float        Percentage of the points considered in the high variance
high_variance_kept = 20           # int          Number of high variance points that go in the training set

'''Create initial empty file from large_pash'''
create_empty_dataset("barrier/large_pash.dat","barrier/large_pash_vide.dat")
data = pash_to_dataframe("barrier/large_pash_vide.dat", features[0], features[1])
train_dataset, test_dataset, \
train_features, train_target, \
test_features, test_target \
    = create_datasets(data, features, "Barrier", frac=frac)
train_features=train_features.reset_index(drop=True)
train_target=train_target.reset_index(drop=True)
train_dataset=train_dataset.reset_index(drop=True)

'''Fill Barrier values for initial training dataset'''
for i in train_features.index:
    epsilon = train_features.loc[i, "epsilon"]
    a3 = train_features.loc[i, "a3"]
    train_target.loc[i] = launch_barrier_type_1(epsilon, a3, i).loc[i,"Barrier"]
    train_dataset.loc[i, "Barrier"] = train_target.loc[i]
print("imported data")

'''Get values from large_pash for mse calculations'''
data_check = pash_to_dataframe("barrier/large_pash.dat", features[0], features[1])

'''Create the Committee'''
committee = Committee(n_models)
normalizer = normalize(train_features)
committee.build_model(normalizer, model_shape, optimizer='adamax')
print('Created models')

'''While the mse on training set is not good enough'''

max_variance = 2*goal_variance
rmse_list = []
epoch_list = [0]

committee.fit(train_features, train_target, epochs=epochs, batch_size=batch_size,
                  verbose=0, bootstrap=bootstrap, split_train=split_train)
list_prediction = committee.predict(data[features])
predicted_target, variance = get_mean_var(list_prediction)
max_variance = np.max(variance)
predicted_target = np.ravel(predicted_target)
rmse = sqrt(calculate_mse(predicted_target, data_check[target]))
rmse_list += [rmse]

while rmse > goal_rmse:
    #Take the top 20% high variance
    variance_dataframe = retransform(data[features + [target]], variance, target_keys=['variance'])
    variance_dataframe = variance_dataframe.sort_values(by='variance', ascending=False)
    variance_dataframe = variance_dataframe.head(int(top_variance_percentage * len(data)))
    variance_dataframe = variance_dataframe[variance_dataframe["variance"] > goal_variance]
    #Create more data
    new_train_features = variance_dataframe.sample(n=min(high_variance_kept, len(variance_dataframe)))[features]
    new_training = pd.DataFrame(columns=features + [target])
    for i in new_train_features.index:
        epsilon = new_train_features.loc[i, "epsilon"]
        a3 = new_train_features.loc[i, "a3"]
        check_exist = train_features[train_features["epsilon"] == epsilon]
        check_exist = check_exist[check_exist["a3"] == a3]
        if len(check_exist) == 0:
            new_training = new_training.append(launch_barrier_type_1(epsilon, a3, i + len(train_dataset)))
    #Append with correct indices
    train_dataset = train_dataset.append(new_training)
    train_dataset = train_dataset.drop_duplicates()
    train_dataset = train_dataset.reset_index(drop=True)
    train_features = train_dataset[features]
    train_target = train_dataset[target]
    #Fit committee
    print("fitting Committee")
    committee.fit(train_features, train_target, epochs=epochs, batch_size=batch_size,
                  verbose=0, bootstrap=bootstrap, split_train=split_train)
    print("fitted Committee")
    #Get new rmse and max_variance
    list_prediction = committee.predict(data[features])
    predicted_target, variance = get_mean_var(list_prediction)
    max_variance = np.max(variance)
    predicted_target = np.ravel(predicted_target)
    rmse = sqrt(calculate_mse(predicted_target, data_check[target]))
    rmse_list += [rmse]
    epoch_list += [epoch_list[-1] + epochs]

'''Compare predicted to a real dataset'''
plt.plot(epoch_list, rmse_list, '+')
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.show()
o = multiple_plots(committee, features, target, data, train_dataset)


