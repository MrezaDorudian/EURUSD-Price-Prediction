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
    future_step: int
        A number indicating that how many
        further candle we want to predict.
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

    def __init__(self, file_name, future_step, method):
        """
        Initialize fields and then call methods to
        fill them with calculated values.
        Attributes
        ----------
        file_name : str
            A string indicating the data file
            without it's extension.
        data: list or ndarray of shape (n, )
            A variable to store the dataset.
        changing_percentage: list
            The prediction will be between
            two elements of changing_percentage.
        classified_data:
        future_step: int
            A number indicating that how many
            further candle we want to predict.

        Returns
        -------
        No return.
        """
        self.future_step = future_step

        self.data = None
        self.changing_percentage = None
        self.classified_data = None
        self.prepare_data(file_name)
        if method == 1:
            self.classify_data_first_method()
        elif method == 2:
            self.classify_data_second_method()
        elif method == 3:
            self.classify_data_third_method()
        elif method == 4:
            self.classify_data_forth_method()


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
        self.data = data
        pass

    def classify_data_first_method(self):
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
        closing_price = self.data['Close'].values
        percentage = []
        for i in range(0, len(closing_price) - self.future_step):
            percentage.append(((closing_price[i + self.future_step] / closing_price[i]) - 1) * 100)

        new_percentage = np.sort(percentage, kind='quicksort')

        chunk_size = round(len(new_percentage) / self.CLASS_NUMBERS)
        divided_data = [new_percentage[i: i + chunk_size] for i in range(0, len(new_percentage), chunk_size)]

        changing_percentage = []
        for i in range(self.CLASS_NUMBERS):
            if i == 0:
                changing_percentage.append(divided_data[i][-1] - divided_data[i][0])
            else:
                changing_percentage.append(divided_data[i][-1] - divided_data[i - 1][-1])

        minimum_change = min(percentage)

        classes = [minimum_change]
        for i in range(5):
            minimum_change += changing_percentage[i]
            classes.append(minimum_change)
        self.changing_percentage = classes

        y_axis = []
        for element in percentage:
            for i in range(len(classes)):
                if classes[i] <= element <= classes[i + 1]:
                    y_axis.append(i)
                    break
        self.classified_data = self.data[:-self.future_step].copy()
        self.classified_data['Class'] = y_axis
        pass

    def classify_data_second_method(self):
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
        closing_price = self.data['Close'].values
        percentage = []
        for i in range(0, len(closing_price) - self.future_step):
            percentage.append(((closing_price[i + self.future_step] / closing_price[i]) - 1) * 100)

        minimum_change = min(percentage)
        change = (max(percentage) - min(percentage)) / 5

        classes = [minimum_change]
        for i in range(5):
            minimum_change += change
            classes.append(minimum_change)
        self.changing_percentage = classes

        y_axis = []
        for element in percentage:
            for i in range(len(classes)):
                if classes[i] <= element <= classes[i + 1]:
                    y_axis.append(i)
                    break
        self.classified_data = self.data[:-self.future_step].copy()
        self.classified_data['Class'] = y_axis
        pass

    def classify_data_third_method(self):
        closing_price = self.data['Close'].values
        percentage = []
        for i in range(0, len(closing_price) - self.future_step):
            percentage.append(((max(closing_price[i:i + self.future_step]) / closing_price[i]) - 1) * 100)

        new_percentage = np.sort(percentage, kind='quicksort')

        chunk_size = round(len(new_percentage) / self.CLASS_NUMBERS)
        divided_data = [new_percentage[i: i + chunk_size] for i in range(0, len(new_percentage), chunk_size)]

        changing_percentage = []
        for i in range(self.CLASS_NUMBERS):
            if i == 0:
                changing_percentage.append(divided_data[i][-1] - divided_data[i][0])
            else:
                changing_percentage.append(divided_data[i][-1] - divided_data[i - 1][-1])

        minimum_change = min(percentage)

        classes = [minimum_change]
        for i in range(5):
            minimum_change += changing_percentage[i]
            classes.append(minimum_change)
        self.changing_percentage = classes

        y_axis = []
        for element in percentage:
            for i in range(len(classes)):
                if classes[i] <= element <= classes[i + 1]:
                    y_axis.append(i)
                    break
        self.classified_data = self.data[:-self.future_step].copy()
        self.classified_data['Class'] = y_axis
        with open('asd.txt', 'w') as f:
            f.write(str(y_axis))

    def classify_data_forth_method(self):
        closing_price = self.data['Close'].values
        percentage = []
        for i in range(0, len(closing_price) - self.future_step):
            percentage.append(((min(closing_price[i:i + self.future_step]) / closing_price[i]) - 1) * 100)

        new_percentage = np.sort(percentage, kind='quicksort')

        chunk_size = round(len(new_percentage) / self.CLASS_NUMBERS)
        divided_data = [new_percentage[i: i + chunk_size] for i in range(0, len(new_percentage), chunk_size)]

        changing_percentage = []
        for i in range(self.CLASS_NUMBERS):
            if i == 0:
                changing_percentage.append(divided_data[i][-1] - divided_data[i][0])
            else:
                changing_percentage.append(divided_data[i][-1] - divided_data[i - 1][-1])

        minimum_change = min(percentage)

        classes = [minimum_change]
        for i in range(5):
            minimum_change += changing_percentage[i]
            classes.append(minimum_change)
        self.changing_percentage = classes

        y_axis = []
        for element in percentage:
            for i in range(len(classes)):
                if classes[i] <= element <= classes[i + 1]:
                    y_axis.append(i)
                    break
        self.classified_data = self.data[:-self.future_step].copy()
        self.classified_data['Class'] = y_axis
        with open('asd.txt', 'w') as f:
            f.write(str(y_axis))


    pass