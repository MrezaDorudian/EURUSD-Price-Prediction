# import numpy as np
# import ClasifyData
# from sklearn.linear_model import LogisticRegression as logReg
#
#
# class LogisticRegression:
#     """
#         A class for implementing classic logistic regression
#         for predicting price of a currency, a stock or ...
#
#         Parameters
#         ----------
#         file_name : str, default=''
#             Name of the data file without it's extension
#             e.g. 'eurusd-data'.
#
#         Attributes
#         ----------
#         LEARNING_DATA_RATIO : float
#             Indicates the percentage of learning data into
#             all data we have, also (1 - LEARNING_DATA_RATIO)
#             shows the percentage of testing data into all data.
#
#         data_x : list or ndarray of shape (n, )
#             This data consists of this attributes:
#             [Close, Lag 1, Lag 2, Lag 3, Lag 4, Lag 5]
#             and it's prepared to give to the model.
#
#         learning_data_x:
#             Just like data_x, but a part of it
#             that is for training. it's size
#             depends on LEARNING_DATA_RATIO.
#
#         testing_data_x:
#             Just like data_x, but a part of it
#             that is for testing. it's size
#             depends on LEARNING_DATA_RATIO.
#
#         data_y : list or ndarray of shape (n, )
#             This data is consist of only one attribute
#             that is the corresponding class of the dta_x.
#
#         learning_data_y: list or ndarray of shape (n, )
#             Just like data_y but a part of it
#             that is for learning. it's size
#             depends on LEARNING_DATA_RATIO.
#
#         testing_data_y: list or ndarray of shape (n, )
#             Just like data_y but a part of it
#             that is for testing. it's size
#             depends on LEARNING_DATA_RATIO.
#
#         predicted_data_y: list or ndarray of shape (n, )
#             The output of the algorithm for data_x,
#             that is the prediction of class number.
#
#         predicted_learning_data_y : list or ndarray of shape (n, )
#             The output of the algorithm for learning_data_x,
#             that is the prediction of class number.
#
#         predicted_testing_data_y : list or ndarray of shape (n, )
#             The output of the algorithm for testing_data_y,
#             that is the prediction of class number.
#
#         error : float
#             Mean square error of algorithm on whole data.
#
#         accuracy : float
#             The number of correct prediction into all of
#             predictions.
#
#         Notes
#         -----
#         This model is better than linear regression
#         and can get maximum accuracy of 60% on the data.
#         it's not much high but acceptable.
#
#
#         References
#         ----------
#         .. [1] https://blog.faradars.org/logistic-regression
#         .. [2] https://www.youtube.com/watch?v=zAULhNrnuL4
#         .. [3] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#
#         Examples
#         --------
#         import LogisticRegression
#         Log_reg = LogisticRegression.LogisticRegression('EURUSD')
#         print(log_reg.error)
#         print(log_reg.accuracy)
#         """
#     LEARNING_DATA_RATIO = 0.8
#
#     def __init__(self, file_name):
#         """
#         Initialize fields and then call methods to
#         fill them with calculated values.
#
#         Parameters
#         ----------
#         file_name : str
#             A string indicating the data file
#             without it's extension.
#
#         Returns
#         -------
#         No return.
#
#         """
#         self.data_x, self.learning_data_x, self.testing_data_x = None, None, None
#         self.data_y, self.learning_data_y, self.testing_data_y = None, None, None
#         self.predicted_data_y, self.predicted_learning_data_y, self.predicted_testing_data_y = None, None, None
#         self.error: float
#         self.accuracy: float
#
#         self.prepare_data(file_name)
#         self.fit()
#         self.calculate_error()
#         self.calculate_accuracy()
#         pass
#
#     def prepare_data(self, file_name):
#         """
#         With the given file_name, this method read data
#         from the file and create 5 lags (older prices)
#         and the price corresponding class, and finally
#         set the data_x, learning_data_x, testing_data_x.
#
#         Parameters
#         ----------
#         file_name : str
#             A string indicating the data file
#             without it's extension.
#
#         Returns
#         -------
#         No return.
#         """
#         classified_data = ClasifyData.DataClassifier(file_name).classified_data
#         classified_data = classified_data.reset_index()
#
#         for i in range(1, 6):
#             classified_data[f'Lag {i}'] = classified_data['Close'].shift(i)
#         classified_data = classified_data.dropna()
#         classified_data = classified_data.drop(['Open', 'High', 'Low'], axis=1)
#         needed_data_x = classified_data[['Close', 'Lag 1', 'Lag 2', 'Lag 3', 'Lag 4', 'Lag 5']]
#         needed_data_y = classified_data[['Class']]
#         x_data = []
#         listed_y_data = []
#         for i in range(len(needed_data_x)):
#             x_data.append(needed_data_x.values.tolist()[i])
#             listed_y_data.append(needed_data_y.values.tolist()[i])
#         y_data = []
#
#         for i in range(len(listed_y_data)):
#             y_data.append(listed_y_data[i][0])
#
#         self.data_x = x_data
#         self.data_y = y_data
#         self.learning_data_x = x_data[:int(np.ceil(len(x_data) * self.LEARNING_DATA_RATIO))]
#         self.learning_data_y = y_data[:int(np.ceil(len(y_data) * self.LEARNING_DATA_RATIO))]
#         self.testing_data_x = x_data[int(np.ceil(len(x_data) * self.LEARNING_DATA_RATIO)):]
#         self.testing_data_y = y_data[int(np.ceil(len(y_data) * self.LEARNING_DATA_RATIO)):]
#         pass
#
#     def fit(self):
#         """
#         Calculate logistic regression and set the
#         predicted_data_y, predicted_learning_data_y
#         and predicted_testing_data_y.
#
#         Parameters
#         ----------
#         No parameter.
#
#         Returns
#         -------
#         No return.
#         """
#         log_reg = logReg(max_iter=500, solver='saga', class_weight='balanced', n_jobs=4)
#         log_reg = log_reg.fit(self.learning_data_x, self.learning_data_y)
#         self.predicted_data_y = log_reg.predict(self.data_x)
#         self.predicted_learning_data_y = self.predicted_data_y[:int(np.ceil(len(self.data_x) * self.LEARNING_DATA_RATIO))]
#         self.predicted_testing_data_y = self.predicted_data_y[int(np.ceil(len(self.data_x) * self.LEARNING_DATA_RATIO)):]
#         pass
#
#     def calculate_error(self):
#         """
#         Calculate error of the model and set
#         it to it's corresponding class field.
#         The Error is 'Mean Square Error'.
#
#         Parameters
#         ----------
#         No parameter
#
#         Returns
#         -------
#         No return.
#
#         """
#         error = 0
#         for i in range(len(self.data_y)):
#             error += np.square(self.data_y[i] - self.predicted_data_y[i])
#         error /= len(self.data_y)
#         self.error = error
#         pass
#
#     def calculate_accuracy(self):
#         """
#         Calculate accuracy of the model and set
#         it to it's corresponding class field.
#         The accuracy is calculated as below:
#         #correct_predicts / #all_predicts.
#
#         Parameters
#         ----------
#         No parameter
#
#         Returns
#         -------
#         No return.
#
#         """
#         point = 0
#         for i in range(len(self.data_y)):
#             if self.data_y[i] == self.predicted_data_y[i]:
#                 point += 1
#         accuracy = point / len(self.data_y)
#         self.accuracy = accuracy
#         pass











import numpy as np
import ClasifyData
from sklearn.linear_model import LogisticRegression as logReg


class LogisticRegression:
    LEARNING_DATA_RATIO = 0.8

    def __init__(self, file_name):
        self.data_x, self.learning_data_x, self.testing_data_x = None, None, None
        self.data_y, self.learning_data_y, self.testing_data_y = None, None, None
        self.predicted_data_y, self.predicted_learning_data_y, self.predicted_testing_data_y = None, None, None
        self.error: float
        self.accuracy: float

        self.prepare_data(file_name)
        self.fit()
        self.calculate_error()
        self.calculate_accuracy()
        pass

    def prepare_data(self, file_name):
        classified_data = ClasifyData.DataClassifier(file_name).classified_data
        classified_data = classified_data.reset_index()

        for i in range(1, 6):
            classified_data[f'Lag {i}'] = classified_data['Close'].shift(i)
        classified_data = classified_data.dropna()
        classified_data = classified_data.drop(['Open', 'High', 'Low'], axis=1)
        needed_data_x = classified_data[['Close', 'Lag 1', 'Lag 2', 'Lag 3', 'Lag 4', 'Lag 5']]
        needed_data_y = classified_data[['Class']]
        x_data = []
        listed_y_data = []
        for i in range(len(needed_data_x)):
            x_data.append(needed_data_x.values.tolist()[i])
            listed_y_data.append(needed_data_y.values.tolist()[i])
        y_data = []

        for i in range(len(listed_y_data)):
            y_data.append(listed_y_data[i][0])

        self.data_x = x_data
        self.data_y = y_data
        self.learning_data_x = x_data[:int(np.ceil(len(x_data) * self.LEARNING_DATA_RATIO))]
        self.learning_data_y = y_data[:int(np.ceil(len(y_data) * self.LEARNING_DATA_RATIO))]
        self.testing_data_x = x_data[int(np.ceil(len(x_data) * self.LEARNING_DATA_RATIO)):]
        self.testing_data_y = y_data[int(np.ceil(len(y_data) * self.LEARNING_DATA_RATIO)):]
        pass

    def fit(self):
        log_reg = logReg(max_iter=500, solver='saga', class_weight='balanced', n_jobs=4)
        log_reg = log_reg.fit(self.learning_data_x, self.learning_data_y)
        self.predicted_data_y = log_reg.predict(self.data_x)
        self.predicted_learning_data_y = self.predicted_data_y[:int(np.ceil(len(self.data_x) * self.LEARNING_DATA_RATIO))]
        self.predicted_testing_data_y = self.predicted_data_y[int(np.ceil(len(self.data_x) * self.LEARNING_DATA_RATIO)):]
        pass

    def calculate_error(self):
        error = 0
        for i in range(len(self.data_y)):
            error += np.square(self.data_y[i] - self.predicted_data_y[i])
        error /= len(self.data_y)
        self.error = error
        pass

    def calculate_accuracy(self):
        point = 0
        for i in range(len(self.data_y)):
            if self.data_y[i] == self.predicted_data_y[i]:
                point += 1
        accuracy = point / len(self.data_y)
        self.accuracy = accuracy
        pass