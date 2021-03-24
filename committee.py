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
    def fit(self, train_features, train_labels, epochs, verbose = 1, split_train = False, bootstrap = None):
        history_list = []
        if split_train:
            n = len(train_features)//self.models_number
        i = 0
        for model in self.models:
            print("train model "+str(i+1)+"/"+str(self.models_number))
            if type(bootstrap) is float:
                train_features_spec = train_features.sample(frac=bootstrap)
                indexes = train_features_spec.index
                train_labels_spec = train_labels[indexes]
            elif split_train:
                train_features_spec = train_features.sample(frac=1./(self.models_number-i))
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
    n, b, p= ax.hist(predicted_list)
    ax.vlines(expected_value, 0, max(n), colors="r")
    ax.vlines(np.mean(predicted_list), 0, max(n), colors="g")
    return 0


class HistOnClick:
    def __init__(self, ax1, ax2, Committee, data, features, target, in_order = True):
        self.Committee = Committee
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax1.figure.canvas.mpl_connect('button_press_event', self.on_click)

        self.in_order = in_order
        self.data = data
        self.features = features
        self.target = target
        if in_order is True:
            index = 0
            diff = 1
            while diff > 0:
                diff = data.loc[index+1, features[1]]-data.loc[index, features[1]]
                index += 1
            lent = len(data)
            self.min1 = data.loc[0,features[0]]
            max1 = data.loc[lent-1, features[0]]
            self.min2 = data.loc[0,features[1]]
            len2 = index
            len1 = lent // index
            self.len2 = len2
            max2 = data.loc[len2 - 1, features[1]]
            self.step1 = (max1 - self.min1 ) / len1
            self.step2 = (max2 - self.min2 ) / len2
    def on_click (self, event):
        features = self.features
        target = self.target
        coord = np.array([event.xdata, event.ydata])
        if self.in_order is False:
            index = 0
            norm = 1000
            for i in range(len(data)):
                point_features = data.loc[[i], features]
                newnorm = np.linalg.norm(point_features-coord)
                if newnorm < norm :
                    norm = newnorm
                    index = i
        else:
            pos1 = round((coord[0] - self.min1 )/self.step1 - 1)
            pos2 = round((coord[1] - self.min2 )/self.step2 - 1)
            index = int(pos1*self.len2 + pos2)
        self.ax2.cla()
        self.ax1.set_data(data.loc[[index], features[0]], data.loc[[index], features[1]])
        point_features = data.loc[[index], features]
        expected_value = data.loc[[index], target]
        plot_histo(Committee, point_features, expected_value, self.ax2)
        self.ax2.figure.canvas.draw()

def multiple_plots(Committee, features, target, data, train_dataset):
    list_prediction = Committee.predict(data[features])
    predicted_target, variance = get_mean_var(list_prediction)
    predicted_target = retransform(data[features], predicted_target)
    variance = retransform(data[features], variance)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    plot_contour(variance, features[0], features[1], target, colorbar=True, ax=ax1, bar_name="Variance")
    plot_contour(predicted_target, features[0], features[1], target, colorbar=True, levels=40, ax=ax3,
                 bar_name="Predicted Barrier")

    diff = np.abs(data[target] - predicted_target[target])
    data_diff = retransform(data[features], diff)
    plot_contour(data_diff, features[0], features[1], target, colorbar=True, ax=ax4, levels=40,
                 bar_name="Barrier difference with expected")

    plot_points(train_dataset, features, ax1)
    plot_points(train_dataset, features, ax3)
    plot_points(train_dataset, features, ax4)

    pointer, = ax1.plot([0], [0], "+")
    object = HistOnClick(pointer, ax2, Committee, data, features, target)

    plt.show()
    return object

if __name__ == "__main__":
    features = ["epsilon", "a3"]
    data = pash_to_dataframe("barrier/large_pash.dat")
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(data, features, "Barrier", frac=0.05)

    Committee = Committee(50)

    normalizer = normalize(train_features)
    Committee.build_model(normalizer, [100, 100, 100])
    Committee.fit(train_features, train_labels, epochs=2000, verbose=0, split_train=False)

    o = multiple_plots(Committee, features, "Barrier", data, train_dataset)

