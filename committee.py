from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface, plot_heatmap, plot_contour, plot_points
import matplotlib.pyplot as plt


class Committee:
    def __init__(self, models_number):
        self.models_number = models_number
        self.models = []
    def build_model(self, normalizer, layers):
        for i in range (self.models_number):
            self.models += [build_model(normalizer, layers)]
    def fit(self, train_features, train_labels, epochs, verbose = 1, split_train = False):
        history_list = []
        if split_train:
            n = len(train_features)//self.models_number
        i = 0
        for model in self.models:
            print("train model "+str(i+1)+"/"+str(self.models_number))
            if split_train:
                train_features_spec = train_features.sample(frac=1/(self.models_number-i))
                indexes = train_features_spec.index
                train_labels_spec = train_labels[indexes]
                train_features = train_features.drop(indexes)
                train_labels = train_labels.drop(indexes)
            else:
                train_features_spec = train_features
                train_labels_spec = train_labels
            history_list += [model.fit(train_features_spec, train_labels_spec, epochs=epochs, verbose=verbose)]
            i += 1
        return history_list
    def retransform (self, features, data):
        predicted_dataset_list = []
        for model in self.models:
            predicted_dataset_list += [retransform(model, features, data)]
        return predicted_dataset_list
    def predict (self, data_features):
        list_prediction = []
        for model in self.models:
            list_prediction += [model.predict(data_features)]
        return list_prediction

def get_mean_var(list_prediction):
    mean_list = np.mean(list_prediction, 0)
    var_list = np.var(list_prediction, 0)
    return mean_list, var_list

def plot_histo (Committee, point_features, expected_value, ax=plt.gca()):
    predicted_list = np.ravel(Committee.predict(point_features))
    n, b, p= plt.hist(predicted_list)
    ax.vlines(expected_value, 0, max(n), colors="r")
    return 0



class HistOnClick:
    def __init__(self, ax1, ax2, Committee, data, features, target):
        self.Committee = Committee
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax1.figure.canvas.mpl_connect('button_press_event', self.on_click)

        self.data = data
        self.features = features
        self.target = target
    def on_click (self, event):
        features = self.features
        target = self.target
        coord = np.array([event.xdata, event.ydata])
        index = 0
        norm = 1000
        for i in range(len(data)):
            point_features = data.loc[[i], features]
            newnorm = np.linalg.norm(point_features-coord)
            if newnorm < norm :
                norm = newnorm
                index = i
        self.ax1.set_data(data.loc[[index], features[0]], data.loc[[index], features[1]])
        point_features = data.loc[[index], features]
        expected_value = data.loc[[index], target]
        self.ax2.cla()
        plot_histo(Committee, point_features, expected_value, self.ax2)
        self.ax2.figure.canvas.draw()



if __name__ == "__main__":
    features = ["epsilon", "a3"]
    data = pash_to_dataframe("barrier/large_pash.dat")
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(data, features, "Barrier", frac=0.1)

    Committee = Committee(100)

    normalizer = normalize(train_features)
    Committee.build_model(normalizer, [100, 100, 100])
    Committee.fit(train_features, train_labels, epochs=1000, verbose=0, split_train=True)
    #plot_histo(Committee, test_dataset.loc[[11],features], test_dataset.loc[[11],"Barrier"])
    #plt.show()
    list_prediction = Committee.predict(data[features])
    predicted_target, variance = get_mean_var(list_prediction)
    predicted_data = retransform(data[features], variance)

    fig, (ax1,ax2)= plt.subplots(1,2)
    pointer, = ax1.plot([0], [0], "+")
    object = HistOnClick(pointer, ax2, Committee, data, features, "Barrier")
    plot_contour(predicted_data, features[0], features[1], "Barrier", colorbar=True, ax=ax1)
    plot_points(train_dataset, features, ax1)
    plt.show()
