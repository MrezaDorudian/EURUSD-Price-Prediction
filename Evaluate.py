import matplotlib.pylab as plt
import numpy as np


class Evaluate:
    def __init__(self, model, history, changing_percentage, x_train, y_train, x_test, y_test, total_data, learning_ratio, time_step):
        self.model = model
        self.history = history
        self.changing_percentage = changing_percentage
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.total_data = total_data
        self.learning_ratio = learning_ratio
        self.time_step = time_step
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None
        self.train_class_accuracy = None
        self.test_class_accuracy = None
        pass

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
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - width / 2, train, width, label='Training set')
        ax.bar(x + width / 2, test, width, label='Testing set')
        ax.set_ylabel('Accuracy(%)')
        ax.set_title('Accuracy by Training and Testing set')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        plt.show()

    def show_info(self):
        print(f'class 1: between {round(self.changing_percentage[0], 5)}% to {round(self.changing_percentage[1], 5)}%.')
        print(f'class 2: between {round(self.changing_percentage[1], 5)}% to {round(self.changing_percentage[2], 5)}%.')
        print(f'class 3: between {round(self.changing_percentage[2], 5)}% to {round(self.changing_percentage[3], 5)}%.')
        print(f'class 4: between {round(self.changing_percentage[3], 5)}% to {round(self.changing_percentage[4], 5)}%.')
        print(f'class 5: between {round(self.changing_percentage[4], 5)}% to {round(self.changing_percentage[5], 5)}%.')
        pass

    def find_peak(self, data_index):
        all_period = self.x_test[data_index - 30 + 1:data_index + 1]
        prediction = self.model.predict(all_period)
        predicted_labels = []
        for candle in prediction:
            predicted_labels.append(np.argmax(candle))
        peak_class = max(predicted_labels)
        valley_class = min(predicted_labels)
        max_output = []
        min_output = []
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == peak_class:
                max_output.append(i + 1)
            if predicted_labels[i] == valley_class:
                min_output.append(i + 1)
        print(f'Peak will probably occur in {max_output[len(max_output) // 2]} minutes from now, and will change between {round(self.changing_percentage[peak_class], 5)}% and {round(self.changing_percentage[peak_class + 1], 5)}%.')
        print(max_output)
        print(f'Valley will probably occur in {min_output[len(min_output) // 2]} minutes from now, and will change between {round(self.changing_percentage[valley_class], 5)}% and {round(self.changing_percentage[valley_class + 1], 5)}%.')
        print(min_output)
        print(f'Price will probably change {round(self.changing_percentage[predicted_labels[-1]], 5)}% to {round(self.changing_percentage[predicted_labels[-1] + 1], 5)}% in 30 minutes from now.')
        dataframe = self.total_data.iloc[int(np.ceil(self.learning_ratio * len(self.total_data))):]
        print(dataframe['Close'].iloc[data_index:data_index + 30].values)
        plt.plot(dataframe['Close'].iloc[data_index:data_index + 30].values)
        plt.show()

    def run(self):
        self.train_loss, self.train_accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        self.test_loss, self.test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.train_class_accuracy = self.calculate_accuracy_for_each_class(self.x_train, self.y_train)
        self.test_class_accuracy = self.calculate_accuracy_for_each_class(self.x_test, self.y_test)

        self.show_accuracy()
        self.show_class_accuracy()
        self.show_info()
        self.find_peak(1566)

