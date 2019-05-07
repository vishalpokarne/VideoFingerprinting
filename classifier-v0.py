import configparser
# import re
import os
import numpy as np
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard, EarlyStopping
from keras.regularizers import l2
import tensorflow

config = configparser.ConfigParser()
config.read_file(open('config.ini', 'r'))


def get_data():
    """

    :return: a numpy ndarray of dimension (m x (n+1)) containing direction data
        loaded from the files, where `m` is the number of data samples and `n`
        is length of direction packets (restricted to 500 to consume less
        computation time and memory). The last column in the data contains the
        class labels of the `m` samples, which are the website numbers.

    This function loads the data from the files and creates a numpy data matrix
    with each row as a data sample and the columns containing packet direction.
    The last column of the data is the label, which is the website to which the
    instance belongs.
    """

    # # modify these parameters in the config file
    # data_path = config.get("paths", "raw_data", 0)      # data folder
    # num_sites = int(config.get('params', 'num_websites', 0))    # 95
    # num_instances = int(config.get('params', 'num_instances', 0))   # 100
    # file_ext = config.get('params', 'file_extension', 0)    # No extension
    # max_length = 500    # maximum number of packet directions to use

    # Removed the 0 as the get function does not take 4 parameters, and 0 is the default value
    # modify these parameters in the config file
    data_path = config.get("paths", "raw_data")  # data folder
    num_videos = int(config.get('params', 'num_videos'))  # 95
    num_instances = int(config.get('params', 'num_instances'))  # 100
    file_ext = config.get('params', 'file_extension')  # No extension
    max_length = 5000  # maximum number of packet directions to use

    # read data from files
    print("loading data...")
    data = []
    for video in range(0, num_videos):
        # print site
        for instance in range(1, num_instances):
            file_name = str(video) + "_" + str(instance)
            # Directory of the raw data
            if os.path.isfile(data_path + file_name + "." + file_ext):
                with open(data_path + file_name + "." + file_ext, encoding="utf8", errors="ignore") as file_pt:
                    # print(file_name + "." + file_ext)
                    directions = []
                    for line in file_pt:
                        line = line.replace("\n", "")
                        line = line.replace("\r", "")
                        try:
                            if line != ",":
                                x = line.split(',')[0].strip()
                                # print(x)
                                if x.isalpha():
                                    # print("alphabets", x)
                                    continue
                                else:
                                    directions.append(float(int(x)))
                        except:
                            pass
                    if len(directions) < max_length:  # if the value is greater than 0, and -1 if its not
                        zend = max_length - len(directions)
                        directions.extend([0] * zend)  # Extend the list by "zend" number of zeroes
                    elif len(directions) > max_length:
                        directions = directions[
                                     :max_length]  # Limit the length to max_length, and truncate elements after the position max_length-1
                    data.append(directions + [video])
    print("done")
    return np.array(data)


def split_data(X, Y, fraction=0.80, balance_dist=False):
    """
    :param X: a numpy ndarray of dimension (m x n) containing data samples
    :param Y: a numpy ndarray of dimension (m x 1) containing labels for X
    :param fraction: a value between 0 and 1, which will be the fraction of
        data split into training and test sets. value of `fraction` will be the
        training data and the rest being test data.
    :param balance_dist: boolean value. The split is performed with ensured
        class balance if the value is true.
    :return: X_train, Y_train, X_test, Y_test

    This function splits the data into training and test datasets.
    """
    X, Y = shuffle(X, Y)
    m, n = X.shape
    split_index = int(round(m * fraction))
    if balance_dist:
        X_train = np.zeros(shape=(split_index, n))
        X_test = np.zeros(shape=(m - split_index, n))
        Y_train = np.zeros(shape=(split_index,))
        Y_test = np.zeros(shape=(m - split_index,))
        labels = np.unique(Y)
        ind1 = 0
        ind2 = 0
        for i in np.arange(labels.size):
            indices = np.where(Y == labels[i])[0]
            split = int(round(len(indices) * fraction))

            X_train[ind1:ind1 + split, :] = X[indices[:split], :]
            X_test[ind2:ind2 + (indices.size - split), :] = X[indices[split:], :]

            Y_train[ind1:ind1 + split] = Y[indices[:split]]
            Y_test[ind2:ind2 + (indices.size - split)] = Y[indices[split:]]

            ind1 += split
            ind2 += indices.size - split
        X_train, Y_train = shuffle(X_train, Y_train)
        X_test, Y_test = shuffle(X_test, Y_test)
        return X_train, Y_train, X_test, Y_test
    return X[:split_index, :], Y[:split_index], \
           X[split_index:, :], Y[split_index:]


class CNN:
    """
    This class contains a CNN architecture to model Website Traffic
    Fingerprinting using the direction information from undefended data
    """

    def __init__(self, num_features, num_classes):
        """
        :param num_features: number of features (columns) in the data (X)
        :param num_classes: number of unique labels in the data (number of
            websites)
        """
        model = Sequential()
        num_filters = [16, 32, 32, 64, 64, 128]
        filter_sizes = [5, 5, 3, 3, 3, 3]
        dense_1 = 256
        l2_lambda = 0.00001
        # increasing the number of filters too much probably obscures the data
        # layer 1
        model.add(Conv1D(num_filters[0], filter_sizes[0]
                         , input_shape=(num_features, 1), padding="same"
                         , activation='relu', kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(num_filters[1], filter_sizes[1], activation='tanh'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(num_filters[0], filter_sizes[0]
                         , padding="same", strides=2
                         , activation='relu', kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(2))

        # layer 2
        model.add(Conv1D(num_filters[2], filter_sizes[0], activation='relu', strides=2
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(num_filters[3], filter_sizes[1], activation='tanh'
                         , kernel_regularizer=l2(l2_lambda)))
        model.add(Conv1D(num_filters[0], filter_sizes[0]
                         , padding="same", activation='relu', kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(2))

        model.add(Flatten())
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dropout(
            0.33))  # Decreasing the dropout rate to an appropriate value provides finer tuning and more accuracy
        model.add(Dense(num_classes, activation='softmax'))

        adam = keras.optimizers.Adam(lr=0.0005, epsilon=None, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model = model
        # print self.model.summary()

    def fit(self, X_train, Y_train, batch_size, epochs, verbose):
        """
        :param X_train: a numpy ndarray of dimension (k x n) containing
            training data
        :param Y_train: a numpy ndarray of dimension (k x 1) containing
            labels for X_train
        :param batch_size: batch size to use for training
        :param epochs: number of epochs for training
        :param verbose: Console print options for training progress.
            0 - silent mode,
            1 - progress bar,
            2 - one line per epoch
        :return: None

        This method starts training the model with the given data. The
        training options are configured with tensorboard and early stopping
        callbacks.

        Tensorboard could be launched by navigating to the directory
        containing this file in terminal and running the following command.
            > tensorboard --logdir graph
        """
        tboard_cb = TensorBoard(log_dir='./graph', histogram_freq=0,
                                write_graph=True, write_images=True)
        early_stopping_cb = EarlyStopping(monitor="val_loss", patience=5)
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs
                       , verbose=verbose, validation_split=0.20
                       , callbacks=[tboard_cb, early_stopping_cb])


def main():
    # Load the data and create X and Y matrices
    data = get_data()
    num_features = data.shape[1] - 1
    X = data[:, :num_features]
    Y = data[:, -1]

    # split the data into training and test set
    X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.98, balance_dist=False)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # instantiate the CNN model and train on the data
    model = CNN(num_features, Y_train.shape[1])
    if X_train.shape[0] < 1000:
        model.fit(X_train, Y_train, batch_size=10, epochs=45, verbose=1)  # Decreasing the batch size improves accuracy
        # Evaluate the trained model on test data and print the accuracy
        score = model.model.evaluate(X_test, Y_test, batch_size=15)
        print("Test accuracy: ", round(score[1] * 100, 2))
        print("Test loss: ", round(score[0], 2))
    else:
        model.fit(X_train, Y_train, batch_size=15, epochs=45, verbose=1)  # Decreasing the batch size improves accuracy
        # Evaluate the trained model on test data and print the accuracy
        score = model.model.evaluate(X_test, Y_test, batch_size=25)
        print("Test accuracy: ", round(score[1] * 100, 2))
        print("Test loss: ", round(score[0], 2))
    # Evaluate model
    # scores = model.model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (model.model.metrics_names[1], scores[1] * 100))

    #Serialize trained model to json
    json_model = model.model.to_json()
    with open("D:\\VideoFingerprint\\SavedModels\\LR0005\\RTR98.json", "w") as json_file:
       json_file.write(json_model)

    # Serialize weights to h5 file
    model.model.save_weights("D:\\VideoFingerprint\\SavedModels\\LR0005\\RTR98.h5")
    print("Saved the model to the disk")


if __name__ == '__main__':
    main()
