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
        self.train_class_accuracy = None
        self.test_class_accuracy = None
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

    def calculate_accuracy_for_each_class(self, x, y):
        predicted = self.model.predict(x)
        predicted_class = []
        actual_class = y

        for p in predicted:
            predicted_class.append(np.argmax(p))
        accuracy = [0, 0, 0, 0, 0]
        counts = [list(actual_class).count(i) for i in range(5)]

        for act, pred in zip(actual_class, predicted_class):
            if act == pred:
                accuracy[act] += 1
        final_accuracy = [i / j for i, j in zip(accuracy, counts)]
        return final_accuracy

    def show_class_accuracy(self):
        labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
        train = self.train_class_accuracy
        test = self.test_class_accuracy

        x = np.arange(len(labels))
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, train, width, label='Training set')
        rects2 = ax.bar(x + width / 2, test, width, label='Testing set')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy(%)')
        ax.set_title('Accuracy by Training and Testing set')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

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
        self.train_loss, self.train_accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        self.test_loss, self.test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.show_accuracy()
        self.train_class_accuracy = self.calculate_accuracy_for_each_class(self.x_train, self.y_train)
        self.test_class_accuracy = self.calculate_accuracy_for_each_class(self.x_test, self.y_test)
        self.show_class_accuracy()
        additional_info = f"""
        class 0: between {self.changing_percentage[0]} to {self.changing_percentage[1]}.
        class 1: between {self.changing_percentage[1]} to {self.changing_percentage[2]}.
        class 2: between {self.changing_percentage[2]} to {self.changing_percentage[3]}.
        class 3: between {self.changing_percentage[3]} to {self.changing_percentage[4]}.
        class 4: between {self.changing_percentage[4]} to {self.changing_percentage[5]}.
        """
        print(additional_info)

