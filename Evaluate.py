import matplotlib.pylab as plt
import numpy as np


class Evaluate:
    def __init__(self, model, history, changing_percentage):
        self.model = model
        self.history = history
        self.changing_percentage = changing_percentage
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
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] += 1
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

    def show_changing_percentage(self):
        print(f'class 1: between {round(self.changing_percentage[0], 5)}% to {round(self.changing_percentage[1], 5)}%.')
        print(f'class 2: between {round(self.changing_percentage[1], 5)}% to {round(self.changing_percentage[2], 5)}%.')
        print(f'class 3: between {round(self.changing_percentage[2], 5)}% to {round(self.changing_percentage[3], 5)}%.')
        print(f'class 4: between {round(self.changing_percentage[3], 5)}% to {round(self.changing_percentage[4], 5)}%.')
        print(f'class 5: between {round(self.changing_percentage[4], 5)}% to {round(self.changing_percentage[5], 5)}%.')

    def show_info(self, maximum_upcoming_min, predicted_label, actual_change, actual_cass, plot_data, peak_class):
        print(f'Peak will probably occur in {maximum_upcoming_min} minutes from now, and will change between {round(self.changing_percentage[peak_class], 5)}% and {round(self.changing_percentage[peak_class + 1], 5)}%.')
        print(f'Price will probably change {round(self.changing_percentage[predicted_label], 5)}% to {round(self.changing_percentage[predicted_label + 1], 5)}% in 30 minutes from now.')
        print(f'Actual change is: {actual_change}')
        print(f'Predicted class is:{predicted_label}, actual class is: {actual_cass}')
        print()
        plt.plot(plot_data)
        plt.show()
        pass

    def predict(self, dataset, x, y, data_index):
        dataset = dataset[30:]
        past_time = dataset.iloc[data_index - 30 + 1]['Date']
        past_price = dataset.iloc[data_index - 30 + 1]['Close']

        current_time = dataset.iloc[data_index]['Date']
        current_price = dataset.iloc[data_index]['Close']

        future_time = dataset.iloc[data_index + 30]['Date']
        future_price = dataset.iloc[data_index + 30]['Close']

        all_period = x[data_index - 30 + 1:data_index + 1]
        prediction = self.model.predict(all_period)
        predicted_labels = []
        for candle in prediction:
            predicted_labels.append(np.argmax(candle))
        peak_class = max(predicted_labels)
        max_output = []
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == peak_class:
                max_output.append(i)
        max_price = []
        max_idx = []
        for idx in max_output:
            max_price.append(dataset['Close'].iloc[idx])
            max_idx.append(idx)

        maximum_upcoming_min = max_output[np.argmax(max_price)]
        actual_change = round(((future_price / current_price) - 1) * 100, 5)
        actual_class = y[data_index]
        plot_data = dataset['Close'].iloc[data_index:data_index + 30].values
        self.show_info(maximum_upcoming_min, predicted_labels[-1], actual_change, actual_class, plot_data, peak_class)

    def run(self, x_train, y_train, x_test, y_test, testing_dataset, scale_data):
        self.train_loss, self.train_accuracy = self.model.evaluate(x_train, y_train, verbose=0)
        self.test_loss, self.test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        self.train_class_accuracy = self.calculate_accuracy_for_each_class(x_train, y_train)
        self.test_class_accuracy = self.calculate_accuracy_for_each_class(x_test, y_test)
        self.show_accuracy()
        self.show_class_accuracy()
        self.show_changing_percentage()
        for i in range(2500, 2530):
            self.predict(testing_dataset, x_test, y_test, i)

