"""
Generates and saves a Committee of neural networks, trained generating data using
the barrier code where the variance of the networks' output is high
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn_regression import normalize, retransform
from committee import Committee, get_mean_var, interactive_plots, save_committee
from hyperparameters import calculate_mse
from launch_barrier import launch_barrier_type_1, change_input, pash_to_dataframe, input_template, launch_barrier

if __name__ == "__main__":
    # |||||||||||||||||||||||   Inputs
    epsilon_range = [-0.1, 0.9, 9]  # [f,f,int]    Range and size of initial dataset creation
    alpha_range = [0, 0.2, 9]       # [f,f,int]    Range and size of initial dataset creation
    empty_grid_size = 101           # int          Variance will be tested on a grid of this size squared
    goal_variance = 0.1             # float        Goal for the maximum variance
    epoch_max = 10000               # int          Epoch at which the training automatically stops
    size_max = 400                  # int          Size maximum of the total train dataset
    features = ["epsilon", "a3"]    # [str]        Key argument of the features
    target = "Barrier"              # str          Key argument of the target
    n_models = 50                   # int          Number of models in the committee
    model_shape = [150, 150, 150]   # [int]        Shape of the neural networks
    optimizer = 'adamax'            # str          Optimizer of the fit
    epochs = 100                    # int          Number of epochs per loop
    batch_size = 10                 # int          Batch size during the fit
    split_train = True              # Bool         Split the train set between the models
    bootstrap = 0.55                # float        Overwrites split_train : bootstraps with this fraction
    top_variance_percentage = 0.2   # float        Percentage of the points considered in the high variance
    high_variance_kept = 20         # int          Number of points that go in the training set each loop
    constant_training_sets = True   # Bool         If True, the training datasets are not
    #                               reshuffled between each iteration between the training sets
    #                               each set is a sample with frac=bootstrap.
    #                               Overwrites bootstrap and split_train
    name_committee = "active_trained7" # Name under which the committee is stored

    '''Create initial empty file from large_pash'''
    # |||||||||||||||||| Create an empty grid:
    alpha3 = np.linspace(alpha_range[0], alpha_range[1], empty_grid_size)
    epsilon = np.linspace(epsilon_range[0], epsilon_range[1], empty_grid_size)
    data_empty = np.array(np.meshgrid(epsilon, alpha3))
    data_empty = data_empty.T.reshape(-1, 2)
    data_empty = pd.DataFrame(data_empty, columns=features)
    # |||||||||||||||||| Create the initial dataset
    input_template('step3')
    change_input(epsilon_range, alpha_range)
    launch_barrier()
    new_train_dataset = pash_to_dataframe("barrier/pash.dat", start_indice=len(data_empty))
    size = len(new_train_dataset)
    if constant_training_sets:
        train = []
        for i in range(n_models):
            train += [new_train_dataset.sample(frac=bootstrap)]
        new_train_dataset=train
    print("Created the initial train dataset.")
    # |||||||||||||||||| Create the committee
    committee = Committee(n_models)
    normalizer = normalize(data_empty)  # Normalize on the empty dataset
    committee.build_model(normalizer, model_shape, optimizer=optimizer)

    print('Created models')

    # |||||||||||||||||| While the max variance is not good enough
    max_variance = 2 * goal_variance
    variance_stat = [[], [], []]
    epoch_list = [0]
    while max_variance > goal_variance and epoch_list[-1] < epoch_max and size < size_max:
        train_dataset = new_train_dataset
        if constant_training_sets:
            train_features = [dataset[features] for dataset in train_dataset]
            train_target = [dataset[target] for dataset in train_dataset]
        else:
            train_features = new_train_dataset[features]
            train_target = new_train_dataset[target]
        # Fit the Committee
        print("Fitting Committee")
        committee.fit(train_features, train_target, epochs=epochs, batch_size=batch_size,
                      verbose=0, bootstrap=bootstrap, split_train=split_train)
        print("Fitted Committee")
        # Get Highest variance point
        list_prediction = committee.predict(data_empty)
        predicted_target, variance = get_mean_var(list_prediction)
        max_variance = np.max(variance)
        epoch_list += [epoch_list[-1] + epochs]
        variance_stat[0] += [np.max(variance)]
        variance_stat[1] += [np.median(variance)]
        variance_stat[2] += [np.mean(variance)]
        print("Maximum variance is ", max_variance)
        # Take the top % high variance points which also are greater than our goal
        variance_dataframe = retransform(data_empty, variance, target_keys=['variance'])
        variance_dataframe = variance_dataframe.sort_values(by='variance', ascending=False)
        variance_dataframe = variance_dataframe.head(int(top_variance_percentage * len(data_empty)))
        variance_dataframe = variance_dataframe[variance_dataframe["variance"] > goal_variance]
        # Generate more training data from these points
        if constant_training_sets:
            print("training sets of size ",[len(set) for set in train_dataset])
        print("Total training set of size ", size)
        new_train_features = variance_dataframe.sample(n=min(high_variance_kept, len(variance_dataframe)))[features]
        new_training = pd.DataFrame(columns=features + [target])

        if constant_training_sets:
            indexes = np.ravel([dataset.index for dataset in train_features])
        else:
            indexes = train_features.index
        # test if the new points are not already inside the training set
        for i in new_train_features.index:
            if not i in indexes:
                epsilon = new_train_features.loc[i, features[0]]
                a3 = new_train_features.loc[i, features[1]]
                new_training = new_training.append(launch_barrier_type_1(epsilon, a3, i))
        size += len(new_training)
        # Append with correct indices
        if constant_training_sets:
            if len(new_training) > 0:
                new_train_dataset = []
                for i in range(n_models):
                    new_train_dataset += [train_dataset[i].append(new_training.sample(frac=bootstrap))]
            else:
                new_train_dataset = train_dataset
        else:
            new_train_dataset = train_dataset.append(new_training)
        save_committee(name_committee + "_temp", committee)


    if constant_training_sets:
        train_dataset2 = train_dataset[0]
        for i in range(1,n_models):
            train_dataset2 = train_dataset2.append(train_dataset[i])
        train_dataset = train_dataset2.drop_duplicates()
    train_dataset.to_csv("data/csv/"+name_committee+".csv")
    save_committee(name_committee, committee)
    '''Compare predicted to a real dataset'''
    data_test = pash_to_dataframe("data/pash/large_pash.dat")
    list_prediction = committee.predict(data_test[features])
    predicted_target, variance = get_mean_var(list_prediction)
    predicted_target = np.ravel(predicted_target)
    rmse = np.sqrt(calculate_mse(predicted_target, data_test[target]))
    print('last rmse is ', rmse)
    print("training set of size ", len(train_dataset))
    print("test data of size", len(data_test))
    print("maximum variance is ", max_variance)

    epoch_list = epoch_list[1:]
    plt.plot(epoch_list, variance_stat[0], '+')
    plt.plot(epoch_list, variance_stat[1], '+')
    plt.plot(epoch_list, variance_stat[2], '+')
    plt.xlabel("epoch")
    plt.ylabel("variance")
    plt.show()

    o = interactive_plots(committee, features, target, data_test, train_dataset)
    print("yeah")
