# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
# import itertools

COUNTER = 0


class DataLayer:


    def __init__(self, data_file_location, learning_ratio, time_step, feature_scaling):
        self.data_file_location: str = data_file_location
        self.learning_ratio: float = learning_ratio
        self.time_step = time_step
        self.feature_scaling = feature_scaling
        self.raw_data = list
        self.training_dataset = tuple
        self.validation_dataset = tuple
        self.testing_dataset = tuple
        self.dataframe = None
        self.feature_scaling_object = None
        pass

    def load_data(self):
        df = pd.read_csv(self.data_file_location)
        self.raw_data = df.iloc[:, 4:5].values
        self.dataframe = df
        pass

    def apply_time_step(self):
        data_x = []
        data_y = []
        for i in range(self.time_step, len(self.raw_data)):
            data_x.append(self.raw_data[i - self.time_step:i, 0])
            data_y.append(self.raw_data[i][0])
        data_x, data_y = np.array(data_x), np.array(data_y)
        data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))

        t_index = int(np.ceil(self.learning_ratio * len(self.raw_data)))
        v_index = t_index + int((len(self.raw_data) - t_index) / 4)

        self.training_dataset = (data_x[:t_index], data_y[:t_index])
        self.validation_dataset = (data_x[t_index:v_index], data_y[t_index:v_index])
        self.testing_dataset = (data_x[v_index:], data_y[v_index:])
        pass

    def apply_feature_scaling(self):
        sc = MinMaxScaler(feature_range=self.feature_scaling)
        scaled_data = sc.fit_transform(self.raw_data)
        self.raw_data = scaled_data
        self.feature_scaling_object = sc
        pass

    def run(self):
        self.load_data()
        self.apply_feature_scaling()
        self.apply_time_step()
        return self.training_dataset, self.validation_dataset, self.testing_dataset, self.dataframe


class ModelLayer:
    def __init__(self, layer_info):
        self.layer_info = layer_info

    def run(self):
        model = Sequential()
        for key, value in self.layer_info.items():
            if key.split('-')[0] == 'LSTM':
                model.add(LSTM(value[0], return_sequences=value[1], activation=value[2], input_shape=(30, 1)))
            elif key.split('-')[0] == 'Dropout':
                model.add(Dropout(value[0]))
            elif key.split('-')[0] == 'Dense':
                model.add(Dense(1, activation='sigmoid'))
        return model


class TrainingLayer:
    def __init__(self, training_data, validation_data, testing_data, model, optimizer, loss, batch_size, epochs, min_max, dataframe, time_step):
        self.x_train, self.y_train = training_data
        self.x_valid, self.y_valid = validation_data
        self.x_test, self.y_test = testing_data

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_max = min_max
        self.dataframe = dataframe
        self.time_step = time_step
        self.history = None
        pass

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        pass

    def fit(self):
        # history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.x_valid, self.y_valid))
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)
        self.model.summary()
        self.history = history.history['loss']
        pass

    def run(self):
        self.compile()
        self.fit()
        return self.model, self.history


class Controller:
    def __init__(self):
        self.data_layer_info = {}
        self.model_layer_info = {}
        self.training_layer_info = {}
        self.data_layer_instances = []
        self.model_layer_instances = []
        self.training_layer_instances = []

    def generate_data_layer_info(self):
        data_file_location = ['data/10000_data.csv']
        learning_ratio = [0.8]
        time_step = [30]
        feature_scaling = [(0, 1)]
        data_dict = {
            'data_file_location': data_file_location,
            'learning_ratio': learning_ratio,
            'time_step': time_step,
            'feature_scaling': feature_scaling
        }
        self.data_layer_info = data_dict
        pass

    def generate_model_layer_info(self):
        layer_counts = [5]
        lstm_neurons = [30, 50]
        dropout_chance = [0.25]

        model_dict = {
            'layer_counts': layer_counts,
            'lstm_neurons': lstm_neurons,
            'dropout_chance': dropout_chance,
        }
        self.model_layer_info = model_dict
        pass

    def generate_training_layer_info(self):
        batch_size = [100]
        epochs = [2]
        optimizer = ['adam']
        loss = ['mse']
        train_dict = {
            'batch_size': batch_size,
            'epochs': epochs,
            'optimizer': optimizer,
            'loss': loss
        }
        self.training_layer_info = train_dict
        pass

    def generate_data_layer_instances(self):
        first_param = self.data_layer_info['data_file_location']
        second_param = self.data_layer_info['learning_ratio']
        third_param = self.data_layer_info['time_step']
        forth_param = self.data_layer_info['feature_scaling']

        all_combinations = [first_param, second_param, third_param, forth_param]
        all_combinations = list(itertools.product(*all_combinations))
        for instance in all_combinations:
            self.data_layer_instances.append(DataLayer(instance[0], instance[1], instance[2], instance[3]))
        pass

    def generate_model_layer_instances(self):
        first_param = self.model_layer_info['layer_counts']
        second_param = self.model_layer_info['lstm_neurons']
        third_param = self.model_layer_info['dropout_chance']
        all_combinations = [first_param, second_param, third_param]
        all_combinations = list(itertools.product(*all_combinations))
        for instance in all_combinations:
            model_dict = {}
            for i in range(instance[0] - 3):
                model_dict[f'LSTM-{i}'] = [instance[1], True, 'relu']
                model_dict[f'Dropout-{i}'] = [instance[2]]
            model_dict[f'LSTM-last'] = [instance[1], False, 'relu']
            model_dict[f'Dropout-last'] = [instance[2]]
            model_dict['Dense'] = [1, 'linear']
            self.model_layer_instances.append(ModelLayer(model_dict))
        pass

    def generate_training_layer_instances(self):
        first_param = self.data_layer_instances
        second_param = self.model_layer_instances
        third_param = self.training_layer_info['optimizer']
        forth_param = self.training_layer_info['loss']
        fifth_param = self.training_layer_info['batch_size']
        sixth_param = self.training_layer_info['epochs']

        all_combinations = [first_param, second_param, third_param, forth_param, fifth_param, sixth_param]
        all_combinations = list(itertools.product(*all_combinations))

        for instance in all_combinations:
            train, valid, test, df = instance[0].run()
            model = instance[1].run()

            self.training_layer_instances.append(TrainingLayer(
                train, valid, test, model,
                instance[2], instance[3],
                instance[4], instance[5],
                instance[0].feature_scaling_object,
                df, instance[0].time_step
            ))

    pass

    def calculate_loss(self, actual, predicted):
        error = 0
        for i in range(len(actual)):
            error += np.square(actual[i] - predicted[i])
        error /= len(actual)
        return error

    def save_results(self, instance, actual, predicted):
        global COUNTER
        loss = self.calculate_loss(actual, predicted)
        plt.plot(actual, color='red', label='Real EUR/USD Stock Price')
        plt.plot(predicted, color='blue', label='Predicted EUR/USD Stock Price')
        plt.title(f'Loss: {loss}')
        plt.legend()
        plt.savefig(f'images/{COUNTER}.png')
        plt.show()
        model = instance.model.layers
        layers = []
        for i in range(len(model)):
            if str(model[i]).count('LSTM') > 0:
                layers.append(f'LSTM: {model[i].units}')
            elif str(model[i]).count('Dropout') > 0:
                layers.append(f'Dropout: {model[i].rate}')
            elif str(model[i]).count('Dense') > 0:
                layers.append(f'Dense: {model[i].units}')

        optimizer = instance.optimizer
        loss = instance.loss
        batch_size = instance.batch_size
        epochs = instance.epochs
        min_max = instance.min_max
        time_step = instance.time_step
        diff = f'================================Number: {COUNTER}================================'

        line = f'{diff}\nmodel: {layers}\noptimizer: {optimizer}\nloss: {loss}\n' \
               f'batch_size: {batch_size}\nepochs: {epochs}\nmin_max: {min_max}\ntime_step: {time_step}\n{diff}\n'
        with open(f'info/{COUNTER}.txt', 'w') as file:
            file.write(line)

        COUNTER += 1
        pass

    def generate_test_data(self, sc, df, time_step):
        dataset_total = df.iloc[:, 4:5]
        dataset_train = df.iloc[:8000, 4:5]
        dataset_test = df.iloc[8000:, 4:5]

        inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values
        inputs = sc.transform(inputs)
        x_test = []
        for i in range(time_step, len(inputs)):
            x_test.append(inputs[i - time_step:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, dataset_test

    def iterate_models(self):
        for instance in self.training_layer_instances:
            x_test, dataset_test = self.generate_test_data(instance.min_max, instance.dataframe, instance.time_step)
            model, history = instance.run()
            predicted_price = model.predict(x_test)
            predicted_price = instance.min_max.inverse_transform(predicted_price)
            actual_price = dataset_test.values
            self.save_results(instance, actual_price, predicted_price)
        pass

    def run(self):
        self.generate_data_layer_info()
        self.generate_model_layer_info()
        self.generate_training_layer_info()
        self.generate_data_layer_instances()
        self.generate_model_layer_instances()
        self.generate_training_layer_instances()
        self.iterate_models()
        pass


# controller = Controller()
# controller.run()