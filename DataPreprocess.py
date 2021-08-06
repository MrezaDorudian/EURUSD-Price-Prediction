import numpy as np
from ClassifyData import DataClassifier
from sklearn.preprocessing import MinMaxScaler


class Data:

    def __init__(self, file_name='fixed_EURUSD', learning_ratio=0.8, shuffle=False, time_step=10):
        self.file_name = file_name
        self.learning_ratio = learning_ratio
        self.shuffle = shuffle
        self.time_step = time_step
        self.scale_data = MinMaxScaler(feature_range=(-1, 1))
        self.total_data = None
        self.learning_data = None
        self.testing_data = None
        self.changing_percentage = []
        self.prepare_data()
        pass

    def time_step_generator(self, train, test):
        x_training = []
        y_training = []

        x_testing = []
        y_testing = []

        for i in range(self.time_step, len(train)):
            x_training.append(train[i - self.time_step:i])
            y_training.append(train[i])

        for i in range(self.time_step, len(test)):
            x_testing.append(test[i - self.time_step:i])
            y_testing.append(test[i])

        x_training = self.scale_data.fit_transform(x_training)
        x_testing = self.scale_data.fit_transform(x_testing)

        x_training, y_training = np.array(x_training), np.array(y_training)
        x_testing, y_testing = np.array(x_testing), np.array(y_testing)

        x_training = np.reshape(x_training, (x_training.shape[0], x_training.shape[1], 1))
        x_testing = np.reshape(x_testing, (x_testing.shape[0], x_testing.shape[1], 1))
        if self.shuffle:
            index = np.random.permutation(len(x_training))
            x_training, y_training = x_training[index], y_training[index]
        self.learning_data = x_training, y_training
        self.testing_data = x_testing, y_testing
        pass

    def prepare_data(self):
        data_classifier = DataClassifier(self.file_name)
        data = data_classifier.classified_data
        self.total_data = data
        self.changing_percentage = data_classifier.changing_percentage

        dataset_train = data.iloc[:int(np.ceil(self.learning_ratio * len(data)))]
        dataset_test = data.iloc[int(np.ceil(self.learning_ratio * len(data))):]

        train = dataset_train['Class'].values
        test = dataset_test['Class'].values
        self.time_step_generator(train.copy(), test.copy())
        pass
    pass