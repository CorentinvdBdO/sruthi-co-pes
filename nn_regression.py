import tensorflow as tf
import numpy as np
from extract_pash import pash_to_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
from launch_barrier import launch_barrier, input_template

def dataframe_to_dataset(dataframe, features, targets):
    """
    Turns a pandas dataframe into a tensorflow dataset
    :param dataframe: pandas dataframe
    :param features: str or list of str out of dataframe's columns
    :param targets: str or list of str out of dataframe's columns
    :return: dataset
    """
    dataset = (tf.data.Dataset.from_tensor_slices((
        dataframe[features].values,
        dataframe[targets].values)))
    return dataset


if __name__ == "__main__":
    print ("***********************************")
    # Get the data
    data = pash_to_dataframe("barrier/pash.dat")
    print(data)
    features = ["P(1)", "P(2)"]
    target = "Barrier"
    dataset = data[features+[target]] # frame_to_dataset(data, features, target)
    # Build train/test datasets: 50% separation
    train_dataset = dataset.sample(frac=0.5, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Barrier')
    test_labels = test_features.pop('Barrier')

    # Normalize the data
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    # Build model
    model = tf.keras.Sequential([normalizer,
        tf.keras.layers.Dense(500, input_dim=2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Fit model
    model.fit(train_features,train_labels, epochs=1000)

    print (model.predict(train_features),train_labels )
    plt.plot (test_labels, model.predict(test_features), ".")
    plt.show()

