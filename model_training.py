import pandas as pd
import numpy as np
import tensorflow as tf
import keras

class RNN:
    
    def data_import(self):
        self.X_train = np.load("data/X_train.npy")
        self.y_train = np.load("data/y_train.npy")
        self.X_test = np.load("data/X_test.npy")
        self.y_test = np.load("data/y_test.npy")
        return 0

    def gpu_config(self):
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        print("Number of GPU: ", len(physical_devices))
        print(physical_devices)

        if len(physical_devices) != 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        return 0
    
    # params editable
    def create_model_architecture(self):
        nn = keras.models.Sequential()

        nn.add(keras.Input(shape=(20, 8)))
        nn.add(keras.layers.LSTM(units=32, return_sequences=True, dropout=0.2))
        nn.add(keras.layers.LSTM(units=64, return_sequences=True, dropout=0.2))
        nn.add(keras.layers.LSTM(units=128, return_sequences=True, dropout=0.2))
        nn.add(keras.layers.LSTM(units=256, return_sequences=True, dropout=0.2))
        nn.add(keras.layers.LSTM(units=512, return_sequences=True, dropout=0.2))
        nn.add(keras.layers.LSTM(units=512, return_sequences=True, dropout=0.2))
        nn.add(keras.layers.LSTM(units=512, dropout=0.1))
        nn.add(keras.layers.Dense(units=2))

        nn.summary()
        nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        self.nn = nn

        return 0
    
    def train_model(self, epochs):
        self.nn.fit(x=self.X_train, y=self.y_train, batch_size=10, epochs=epochs, verbose=2)
        return 0
    
    def test_model(self):
        predictions = self.nn.predict(x=self.X_test, batch_size=10, verbose=0)
        predictions
        pred = np.argmax(predictions, axis=-1)

        accuracy = 0
        for i in range(len(pred)):
            if pred[i] == self.y_test[i]:
                accuracy = accuracy + 1
        
        accuracy = accuracy/len(self.y_test)
        self.accuracy = int(accuracy * 10000) / 100
        return accuracy
    
    def predict(self, X_test, y_test=0):
        predictions = self.nn.predict(x=X_test, batch_size=10, verbose=0)
        predictions
        pred = np.argmax(predictions, axis=-1)

        if y_test == 0:
            return pred
        
        accuracy = 0
        for i in range(len(pred)):
            if pred[i] == y_test[i]:
                accuracy = accuracy + 1
        
        return accuracy/len(y_test)
    
    def save_model(self, path):
        self.nn.save(path)
        return 0
    
    def auto_train(self):
        self.data_import()
        self.gpu_config()
        self.create_model_architecture()
        self.train_model(10)
        print("Accuracy : ", self.test_model())
        self.save_model(f"data/rnn_{self.accuracy}.h5")

if __name__ == "__main__":
    rnn = RNN()
    rnn.auto_train()