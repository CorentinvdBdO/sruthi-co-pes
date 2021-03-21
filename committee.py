from nn_regression import create_datasets, normalize, build_model, learning_curve, retransform
import numpy as np
from extract_pash import pash_to_dataframe, plot_surface


class Committee:
    def __init__(self, models_number):
        self.models_number = models_number
        self.models = []
    def build_model(self, normalizer, layers):
        for i in range (self.models_number):
            self.models += [build_model(normalizer, layers)]
    def fit(self, train_features, train_labels, epochs, verbose = 1):
        history_list = []
        for model in self.models:
            history_list += [model.fit(train_features, train_labels, epochs, verbose)]
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

if __name__ == "__main__":
    dataset = pash_to_dataframe("barrier/pash_step3new.dat")
    train_dataset, test_dataset, \
    train_features, train_labels, \
    test_features, test_labels \
        = create_datasets(dataset, ["P(1)", "P(2)"], "Barrier")

    Committee = Committee(5)

    normalizer = normalize(train_features)
    Committee.build_model(normalizer, [100,100,100])
    Committee.fit(train_features, train_labels, 500)
    list_prediction = Committee.predict(test_features)
    print(np.mean(list_prediction)