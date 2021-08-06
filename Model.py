from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional

class Model:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.history = None
        self.model = None
        self.build_model()
        self.compile()
        self.fit()
        pass

    def build_model(self):
        self.model = Sequential([
                Bidirectional(LSTM(128,  input_shape=(self.x_train.shape[1], 1), return_sequences=False)),
                Dense(32),
                Dense(5, activation='softmax')
            ])
        pass

    def compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        pass

    def fit(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=128, verbose=1,
                                      validation_data=(self.x_test, self.y_test))
        self.model.summary()
        pass
