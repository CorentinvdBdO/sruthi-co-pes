from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface, plot_heatmap, plot_contour, plot_points
import matplotlib.pyplot as plt
from launch_barrier import launch_barrier, change_file_name, input_template
from committee import Committee
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
    goal_variance = 100
    min_epsilon = 0
    max_epsilon = .95
    min_a3 = 0
    max_a3 = .4
    initial_size = [10,10]
    features = ["epsilon", "a3"]
    target = "Barrier"
    n_models = 5
    model_shape = [150, 150, 150]
    epochs = 1500
    split_train = True
    bootstrap = 0.8
    # |||||||||||||||||      Create initial training data
    input_template('step3')
    change_input([min_epsilon, max_epsilon, initial_size[0]], [min_a3, max_a3, initial_size[1]])
    launch_barrier()
    data_train = pash_to_dataframe("barrier/pash.dat", features[0], features[1])
    train_features = data_train[features]
    train_target = data_train[target]
    # |||||||||||||||||      Create the Committee
    committee = Committee(n_models)
    normalizer = normalize(train_features)
    Committee.build_model(normalizer, model_shape)
    # |||||||||||||||||       While the min variance is not good enough:
    min_variance = 2*goal_variance
    while min_variance > goal_variance:
        # |||||||||||||||||   Fit the Committee
        Committee.fit(train_features, train_target, epochs=epochs, verbose=0, bootstrap=bootstrap, split_train=split_train)
        # |||||||||||||||||   Get Highest variance point
        list_prediction = Committee.predict(train_features)
        predicted_target, variance = get_mean_var(list_prediction)
        variance = retransform(train_features, variance)
        # |||||||||||||||||   Create more data
        launch_barrier()
        # |||||||||||||||||   Append correct indices
    # |||||||||||||||||       Compare predicted to a real dataset

    print("yeah")