import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    """
    A class for implementing classic linear regression
    for predicting price of a currency, a stock or ...

    Parameters
    ----------
    file_name : str, default=''
        Name of the data file without it's extension
        e.g. 'eurusd-data'.

    Attributes
    ----------
    LEARNING_DATA_RATIO : float
        Indicates the percentage of learning data into
        all data we have, also (1 - LEARNING_DATA_RATIO)
        shows the percentage of testing data into all data.

    data : list or ndarray of shape (n, )
        Whole data that read from the input file.

    learning_data : list or ndarray of shape (n, )
        Some part of the whole data that is prepared as
        training data and it's size depends on LEARNING_DATA_RATIO.

    testing_data : list or ndarray of shape (n, )
        Some part of the whole data that is prepared as
        testing data and it's size depends on LEARNING_DATA_RATIO.

    predicted_data: list or ndarray of shape (n, )
        The output of algorithm for data.

    predicted_learning_data : list or ndarray of shape (n, )
        The output of algorithm for learning_data.

    predicted_testing_data : list or ndarray of shape (n, )
        The output of algorithm for testing_data.

    b0 : float
        The constant coefficient of regression.

    b1 : float
        Variable coefficient of regression.

    error : float
        Mean square error of algorithm on whole data.

    Notes
    -----
    This model is not reliable because it can only
    able to draw a line based on the data and it's
    calculated coefficients, but the actual price
    is not even close to a linear function.

    References
    ----------
    .. [1] https://blog.faradars.org/simple-linear-regression
    .. [2] https://www.youtube.com/watch?v=ZkjP5RJLQF4

    Examples
    --------
    import LinearRegression
    linear_regression = LinearRegression('EURUSD')
    print(linear_regression.error)
    """
    LEARNING_DATA_RATIO = 0.8

    def __init__(self, file_name):
        """
        Initialize fields and then call methods to
        fill them with calculated values.

        Parameters
        ----------
        file_name : str
            A string indicating the data file
            without it's extension.

        Returns
        -------
        No return.
        """
        self.data, self.learning_data, self.testing_data = None, None, None
        self.predicted_data, self.predicted_learning_data, self.predicted_testing_data = None, None, None
        self.b0: float
        self.b1: float
        self.error: float

        self.prepare_data(file_name)
        self.fit()
        self.predict()
        self.calculate_error()
        self.plot()
        pass

    def prepare_data(self, file_name):
        """
        With the given file_name, this method read data
        from the file and fetch only 'Close' price of
        data and save it to a class field.

        Parameters
        ----------
        file_name : str
            A string indicating the data file
            without it's extension.

        Returns
        -------
        No return.
        """
        data = pd.read_csv(f'data/{file_name}.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        self.data = data.set_index('Date')['Close'].to_list()
        self.learning_data = data[:int(np.ceil(len(data) * self.LEARNING_DATA_RATIO))]
        self.testing_data = data[int(np.ceil(len(data) * self.LEARNING_DATA_RATIO)):]
        pass

    def fit(self):
        """
        Calculate regression and set the coefficients
        to the class fields.

        Parameters
        ----------
        No parameter.

        Returns
        -------
        No return.
        """
        x_axis = [i for i in range(0, len(self.learning_data))]
        y_axis = self.learning_data['Close'].to_list()
        average_x = sum(x_axis) / len(x_axis)
        average_y = sum(y_axis) / len(y_axis)
        b1_numerator = 0
        b1_denominator = 0
        for i in range(len(y_axis)):
            b1_numerator += (y_axis[i] - average_y) * (x_axis[i] - average_x)
            b1_denominator += np.square(x_axis[i] - average_x)
        b1 = b1_numerator / b1_denominator
        b0 = average_y - b1 * average_x
        self.b0 = b0
        self.b1 = b1
        pass

    def predict(self):
        """
        Predict the data based on the calculated
        coefficients.
        Parameters
        ----------
        No parameter.

        Returns
        -------
        No return.
        """
        predicted_price = []
        for i in range(len(self.data)):
            predicted_price.append(self.b0 + (self.b1 * i))
        self.predicted_data = predicted_price
        self.predicted_learning_data = predicted_price[:int(np.ceil(len(predicted_price) * self.LEARNING_DATA_RATIO))]
        self.predicted_testing_data = predicted_price[int(np.ceil(len(predicted_price) * self.LEARNING_DATA_RATIO)):]
        pass

    def calculate_error(self):
        """
        Calculate error of the model and set
        it to it's corresponding class field.
        The Error is 'Mean Square Error'.

        Parameters
        ----------
        No parameter

        Returns
        -------
        No return.

        """
        error = 0
        for i in range(len(self.data)):
            error += np.square(self.data[i] - self.predicted_data[i])
        error /= len(self.data)
        self.error = error
        pass

    def plot(self):
        """
        Plot the actual and the predicted
        price to understand better what the
        model predicts.

        Parameters
        ----------
        No parameter.

        Returns
        -------
        No return.
        """
        plt.plot(self.predicted_data, c='red', label='Predicted Price')
        plt.plot(self.data, c='blue', label='Actual Price')
        plt.title('actual vs. predicted price using linear regression')
        plt.xlabel('Minutes')
        plt.ylabel('price($)')
        plt.legend()
        plt.show()
        pass
