from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface, plot_heatmap, plot_contour, plot_points
import matplotlib.pyplot as plt
from launch_barrier import launch_barrier, change_file_name, input_template
from committee import Committee, get_mean_var, multiple_plots
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
    goal_variance = 3
    min_epsilon = 0
    max_epsilon = .95
    min_a3 = 0
    max_a3 = .4
    initial_size = [10,10]
    features = ["epsilon", "a3"]
    target = "Barrier"
    frac = 100
    n_models = 5
    model_shape = [150, 150, 150]
    epochs = 500
    split_train = True
    bootstrap = 0.8
    top_variance_percentage = 0.2
    high_variance_kept = 0.02
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
    committee.build_model(normalizer, model_shape)
    print("Created models")
    # |||||||||||||||||       While the min variance is not good enough:
    max_variance = 2*goal_variance
    while max_variance > goal_variance:
        # |||||||||||||||||   Fit the Committee
        print("fitting Committee")
        committee.fit(train_features, train_target, epochs=epochs,
                      verbose=0, bootstrap=bootstrap, split_train=split_train)
        print("fitted Committee")
        # |||||||||||||||||   Get Highest variance point
        list_prediction = committee.predict(data[features])
        predicted_target, variance = get_mean_var(list_prediction)
        max_variance = np.max(variance)
        print("maximum variance is ", max_variance)
        # Take the top 20% high variance
        variance_dataframe = retransform(data[features+[target]], variance, target_keys=['variance'])
        variance_dataframe = variance_dataframe.sort_values(by='variance',ascending = False)
        variance_dataframe = variance_dataframe.head(int(top_variance_percentage*len(data)))
        # |||||||||||||||||   Create more data
        new_training = variance_dataframe.sample(frac=high_variance_kept)[features+[target]]
        # |||||||||||||||||   Append correct indices
        train_dataset = train_dataset.append(new_training)
        train_dataset = train_dataset.drop_duplicates()
        train_features = train_dataset[features]
        train_target = train_dataset[target]
    # |||||||||||||||||       Compare predicted to a real dataset
    o = multiple_plots(committee, features, target, data, train_dataset)
    print("yeah")