import matplotlib.pylab as plt
import numpy as np

class Evaluate:
    def __init__(self, model, history, changing_percentage, x_train, y_train, x_test, y_test, total_data, time_step):
        self.model = model
        self.history = history
        self.changing_percentage = changing_percentage
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.total_data = total_data
        self.time_step = time_step
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None
        pass

    def calculate_changing(self, class_number):
        for i in range(len(self.changing_percentage) - 1):
            if class_number == i:
                return [self.changing_percentage[i], self.changing_percentage[i + 1]]

    def show_accuracy(self):
        plt.plot(self.history.history['accuracy'], label='train accuracy', color='red')
        plt.plot(self.history.history['val_accuracy'], label='validation accuracy', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend()
        plt.show()

    def prediction(self):
        predicted = np.argmax(self.model.predict(self.x_train[0:1]))
        actual = self.y_train[0]
        print(predicted, actual)
        starting, ending = self.calculate_changing(predicted)
        print(f'Price in the next 30 mins will change between {round(starting, 5)}% to {round(ending, 5)}% from now.')
        change = (((self.total_data.iloc[61]['Close'] / self.total_data.iloc[31]['Close']) - 1) * 100)
        print(f'actual change is: {round(change, 5)}')
    def run(self):
        self.train_loss, self.train_accuracy = self.model.evaluate(self.x_train, self.y_train)
        self.test_loss, self.test_accuracy = self.model.evaluate(self.x_test, self.y_test)
        self.prediction()
        self.show_accuracy()

