from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface, plot_heatmap, plot_contour, plot_points
import matplotlib.pyplot as plt
from launch_barrier import launch_barrier, change_file_name, input_template
from committee import Committee, get_mean_var, multiple_plots
from hyperparameters import calculate_mse


def change_input(epsilon, alpha3):
    """
    :param epsilon: list : [min, max, n]
    :param alpha3: list : [min, max, n]
    :return:
    """
    epsilon_step = (epsilon[1]-epsilon[0])/epsilon[2]
    alpha3_step = (alpha3[1]-alpha3[0])/alpha3[2]

    f = open("barrier/barrier.inp", "r")
    lines = f.readlines()
    f.close()
    lines[3]='   vareps=.t.,beg1v={:.4f},step1v={:.4f},end1v={:.4f},\n'.format(epsilon[0],epsilon_step, epsilon[1])
    lines[4]='   varp3=.t., beg2v={:.4f},step2v={:.4f},end2v={:.4f} / END\n'.format(alpha3[0],alpha3_step, alpha3[1])
    f = open("barrier/barrier.inp", "w")
    f.write("".join(lines))
    f.close()

if __name__ == "__main__":
    # |||||||||||||||||   Inputs:
    goal_variance = 0.8               # float        Goal for the maximum variance
    features = ["epsilon", "a3"]      # [str]        Key argument of the features
    target = "Barrier"                # str          Key argument of the target
    frac = 120                        # int or float fraction of the large pash turning into initial training data
    n_models = 10                      # int          Number of models in the committee
    model_shape = [150, 150, 150]     #  Shape of the neural networks
    epochs = 200                      # int          Number of epochs per loop
    batch_size = 30                   # int          Batch size during the fit
    split_train = True                # Bool         Split the train set between the models
    bootstrap = 0.5                   # float        Overwrites split_train : bootstraps with this fraction
    top_variance_percentage = 0.2     # float        Percentage of the points considered in the high variance
    high_variance_kept = 20           # int          Number of high variance points that go in the training set
    # |||||||||||||||||      Create initial training data
    # Here, we cheat and use the precomputed 4000 large dataset
    #input_template('step3')
    #change_input([min_epsilon, max_epsilon, initial_size[0]], [min_a3, max_a3, initial_size[1]])
    #launch_barrier()
    data = pash_to_dataframe("barrier/large_pash.dat", features[0], features[1])
    train_dataset, test_dataset, \
    train_features, train_target, \
    test_features, test_target \
        = create_datasets(data, features, "Barrier", frac=frac)
    print("imported data")
    # |||||||||||||||||      Create the Committee
    committee = Committee(n_models)
    normalizer = normalize(train_features)
    committee.build_model(normalizer, model_shape, optimizer='adamax')
    print('Created models')
    # |||||||||||||||||       While the min variance is not good enough:
    max_variance = 2*goal_variance
    variance_stat=[[],[],[]]
    mse_list = []
    epoch_list = [0]
    while max_variance > goal_variance and epoch_list[-1] < 6000:
        # |||||||||||||||||   Fit the Committee
        print("fitting Committee")
        committee.fit(train_features, train_target, epochs=epochs, batch_size=batch_size,
                      verbose=0, bootstrap=bootstrap, split_train=split_train)
        print("fitted Committee")
        # |||||||||||||||||   Get Highest variance point
        list_prediction = committee.predict(data[features])
        predicted_target, variance = get_mean_var(list_prediction)
        max_variance = np.max(variance)
        predicted_target = np.ravel(predicted_target)
        mse = calculate_mse(predicted_target, data[target])
        mse_list += [mse]
        epoch_list += [epoch_list[-1] + epochs]
        #plt.hist(np.ravel(variance),bins=20)
        #plt.yscale('log')
        #plt.show()
        variance_stat[0] += [np.max(variance)]
        variance_stat[1] += [np.median(variance)]
        variance_stat[2] += [np.mean(variance)]
        print("maximum variance is ", max_variance)
        print("mse is ", mse)
        # Take the top 20% high variance
        variance_dataframe = retransform(data[features+[target]], variance, target_keys=['variance'])
        variance_dataframe = variance_dataframe.sort_values(by='variance',ascending = False)
        variance_dataframe = variance_dataframe.head(int(top_variance_percentage*len(data)))
        variance_dataframe = variance_dataframe[variance_dataframe["variance"]>goal_variance]
        # |||||||||||||||||   Create more data
        print("training set of size ", len(train_dataset))
        new_training = variance_dataframe.sample(n=min(high_variance_kept, len(variance_dataframe)))[features+[target]]
        # |||||||||||||||||   Append with correct indices
        train_dataset = train_dataset.append(new_training)
        train_dataset = train_dataset.drop_duplicates()
        train_features = train_dataset[features]
        train_target = train_dataset[target]
    # |||||||||||||||||       Compare predicted to a real dataset

    epoch_list = epoch_list[1:]

    plt.plot(epoch_list, mse_list, '+')
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.show()

    plt.plot(epoch_list, variance_stat[0], '+')
    plt.plot(epoch_list, variance_stat[1], '+')
    plt.plot(epoch_list, variance_stat[2], '+')
    plt.xlabel("epoch")
    plt.ylabel("variance")
    plt.yscale('log')
    plt.show()

    o = multiple_plots(committee, features, target, data, train_dataset)

    print("yeah")