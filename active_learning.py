from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface, plot_heatmap, plot_contour, plot_points
import matplotlib.pyplot as plt

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
    change_input([0.,0.9,100],[-0.2,0.4,100])
    print("yeah")