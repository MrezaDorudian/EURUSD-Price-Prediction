import numpy as np
import pandas as pd


class DataClassifier:
    """
    A class for Classify the data that we have
    to 5 periods, that each period is likely
    to have equal data in it, but the period
    length can be unequal, because we wanted
    to have equal data in each period.
    Parameters
    ----------
    file_name : str, default=''
        Name of the data file without it's extension
        e.g. 'eurusd-data'.
    Attributes
    ----------
    CLASS_NUMBERS : int
        Indicates the number of periods or
        classes we want to classify the data.
    data : list or ndarray of shape (n, )
        This data consists of this attributes:
        [Open, High, Low, Close]
        and it's prepared to give to the methods
        to classify.
    changing_percentage : list or ndarray of shape (n, )
        A list consists of changing percentages
        and the data must be only in one
        of this changes.
    Notes
    -----
    At the beginning, we sort the data by
    it's closing price and then we divide it
    into 5 chunks. the we calculate each chunk
    changing percentage (max_price / min_price)
    in each chunk.
    In the next step we iterate data again
    and check which class we have to set that
    row of data.
    Examples
    --------
    import ClassifyData
    classified_data = ClassifyData.DataClassifier('EURUSD').classified_data
    """
    CLASS_NUMBERS = 5

    def __init__(self, file_name):
        """
        Initialize fields and then call methods to
        fill them with calculated values.
        Attributes
        ----------
        file_name : str
            A string indicating the data file
            without it's extension.
        Returns
        -------
        No return.
        """
        self.data = None
        self.changing_percentage = None
        self.classified_data = None

        self.prepare_data(file_name)
        self.divide_into_periods()
        self.classify_data()

    def prepare_data(self, file_name):
        """
        With the given file_name, this method read data
        from the file and set it to the class field.
        Parameters
        ----------
        file_name : str
            A string indicating the data file
            without it's extension.
        Returns
        -------
        No return.
        """
        data = pd.read_csv(f'{file_name}.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        self.data = data.set_index('Date')
        pass

    def divide_into_periods(self):
        """
        Find the changing_percentages form
        the given data and the output is
        the percentages that each of our
        data should be in between 2 of
        each percentages.
        Parameters
        ----------
        No parameter.
        Returns
        -------
        No return.
        """
        closing_price = np.sort(self.data['Close'], kind='quicksort')
        chunk_size = round(len(closing_price) / self.CLASS_NUMBERS)
        divided_data = [closing_price[i: i + chunk_size] for i in range(0, len(closing_price), chunk_size)]
        changing_percentage = [0]
        for i in range(self.CLASS_NUMBERS):
            if i == 0:
                changing_percentage.append(((divided_data[i][-1] / divided_data[i][0]) - 1) * 100)
            else:
                changing_percentage.append((((divided_data[i][-1] / divided_data[i - 1][-1]) - 1) * 100))

        # calculate cumulative percentage
        for i in range(1, len(changing_percentage)):
            changing_percentage[i] += changing_percentage[i - 1]
        changing_percentage[-1] = (((closing_price[-1] / closing_price[0]) - 1) * 100)
        self.changing_percentage = changing_percentage
        pass

    def classify_data(self):
        """
        Iterate the data, and check where
        we should put the data based on the
        changing percentage that we calculated
        and set a number for each data as class.
        Parameters
        ----------
        No parameter.
        Returns
        -------
        No return.
        """
        y_axis = []
        for data in range(len(self.data)):
            current_changing_percentage = (self.data['Close'][data] / min(self.data['Close']) - 1) * 100
            for index in range(self.CLASS_NUMBERS):
                if self.changing_percentage[index] <= current_changing_percentage <= self.changing_percentage[index + 1]:
                    y_axis.append(index)
                    break

        self.classified_data = self.data
        self.classified_data['Class'] = y_axis
        pass