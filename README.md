# Sruthi Co PES

Simulating the potential energy surface of a nucleus requires heavy calculations.
The present code aims at using active learning in order to reduce the number of simulations,
thereby reducing the computation time.

## Authors

Sruthiranjani Ravikularaman

Corentin van den Broek d'Obrenan

## Principle

Some points are asked to be simulated by the algorithm
and the rest of the surface is computed using a committee of neural networks.

## Dependencies

os
shutil
math
numpy
matplotlib.pyplot
scipy
pandas
tensorflow.keras

## File organisation

src : source files
src/launch_barrier.py
src/nn_regression.py
src/hyperparameters.py
src/committee.py
src/active_learning.py
src/analysis.py
src/final_analysis.py

barrier : files necessary to run barrier code
barrier/input_templates : a list of templates that can be used
barrier/barrier.exe : barrier code
barrier/barrier.inp : input for the barrier code
barrier/pash.dat : results of a barrier run

data : storage unit for models and generated points
data/csv : DataFrames (epsilon, a3, "Barrier")
data/models : committees and keras models
data/pash : pash.dat files with generated points

results : results of the project
results/hyperparameters : plots of loss as a function of hyperparameters
                          and learning curves generated to optimize the hyperparameters

doc : bibliography, subject of the project, presentation and documentation

## Usage

src/launch_barrier.py : Functions to change the input and run the barrier code and store its data output.
src/nn_regression.py : Functions to manipulate DataFrames for a keras fit,
                       to build models quickly and work on the learning curves.
src/hyperparameters.py : Function to compare hyperparameters.
src/committee.py : Functions regarding the class Committee, managing a committee of models.
src/active_learning.py :
src/analysis.py : Functions to simplify generations of different kinds of plots.
src/final_analysis.py : This file simply loads a committee to study its results.

You will find in src/active_learning.py a code ready to use with many inputs to choose from.
It generates a committee of neural networks and then trains itself generating data from the barrier code.
The committee can then be accessed using Keras' models/load_model as shown in final_analysis.py

The inputs allows for the customization of many aspects, such as the initial training dataset,
the frame from which the new points are generated, the hyperparameters of the committee's neural
networks and of the training, the way in which the new points are chosen and how the training
data is split between the committee's models.

The initial training dataset is generated and simulated by a grid in the epsilon, alpha3 space, defined by
*epsilon_range* and *alpha_range*. The new points are then generated but not yet simulated from a grid using the same
min and max values of epsilon and alpha, but many more steps (*empty_grid_size* steps).
The training stops when the maximal variance over this grid reaches *goal_variance* or when
the total number of epochs reaches *epoch_max*.
A committee of *n_models* neural networks of shape *model_shape* is generated and compiled
using the optimizer *optimizer*.
During each loop, the committee is fitted for *epochs* epochs and each model gets either the
same training set, a different part of the training set if *split_train* is True, a sample of
the training set of fraction *bootstrap* if it is a float, or, when *constant_training_sets* is
True, a dedicated train_dataset that stays the same over each loop, starting as a sample of the
initial train_dataset of fraction *bootstrap*. After the fit, a prediction of the barrier is
asked to each model of the committee. From the points where the variance of this prediction
is high (top *top_variance_percentage*), *high_variance_kept* are computed and join the training
dataset. If *constant_training_sets* is True, a fraction of them join each model's dedicated training
dataset instead.

The training dataset and the committee are then saved in data/csv and data/models to be reused and analysed.
The committee is saved in a temporary file at every iteration, so you can access it even if you stop the computations.
