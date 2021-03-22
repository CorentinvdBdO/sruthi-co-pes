from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface, plot_heatmap
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

def get_histo (Committee, point_features):



if __name__ == "__main__":
    features = ["epsilon", "a3"]
    dataset = pash_to_dataframe("barrier/large_pash.dat")
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(dataset, features, "Barrier", frac=0.05)

    Committee = Committee(5)

    normalizer = normalize(train_features)
    Committee.build_model(normalizer, [100,100,100])
    Committee.fit(train_features, train_labels, epochs=2000, verbose=0, split_train=True)
    list_prediction = Committee.predict(dataset[features])
    predicted_target, variance = get_mean_var(list_prediction)
    predicted_dataset = retransform(dataset[features], variance)
    plot_heatmap(predicted_dataset, features[0], features[1], "Barrier")
    plt.show()
