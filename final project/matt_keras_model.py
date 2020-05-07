import keras
from matt_generate_data import save_distilled_y
import matplotlib.pyplot as plt
import numpy as np


# DEFINE MODEL
def create_model(LABEL_FORMAT, IMAGE_WIDTH, IMAGE_HEIGHT, TEMPERATURE):
    model = keras.Sequential()  # Must define the input shape in the first layer of the neural network
    model.add(
        keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))

    # temperature added
    if LABEL_FORMAT == 'normal':
        T = TEMPERATURE
        model.add(keras.layers.Lambda(lambda x: x / T))
    model.add(keras.layers.Dense(3, activation='softmax'))

    return model


def create_model_api(IMAGE_WIDTH, IMAGE_HEIGHT):
    from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, Activation, BatchNormalization
    from keras.models import Model

    def res_block(input_layer, filters, depth=1):
        shortcut = input_layer
        output_layer=input_layer
        for i in range(depth):
            output_layer = Conv2D(filters=filters, kernel_size=2, padding='same')(output_layer)
            output_layer = BatchNormalization(axis=3)(output_layer)
            if i+1 < depth: # wait to do the last relu activation until after concatenation
                output_layer = Activation('relu')(output_layer)
        output_layer = Concatenate(axis=-1)([output_layer, shortcut])
        output_layer = Activation('relu')(output_layer)
        return output_layer

    inputs = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    conv1 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(inputs)
    maxpool1 = MaxPooling2D(pool_size=2)(conv1)
    dropout1 = Dropout(0.3)(maxpool1) # to skip until after next conv layer
    conv2 = res_block(dropout1, 32, 1)
    maxpool2 = MaxPooling2D(pool_size=2)(conv2)
    dropout2 = Dropout(0.3)(maxpool2)
    conv3 = res_block(dropout2, 16, 1)
    maxpool3 = MaxPooling2D(pool_size=2)(conv3)
    conv4 = res_block(maxpool3, 8, 1)
    maxpool4 = MaxPooling2D(pool_size=2)(conv4)
    flatten = Flatten()(maxpool4)
    dense1 = Dense(256, activation='relu')(flatten)
    outputs = Dense(3, activation='softmax')(dense1)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def save_weights(model, LABEL_FORMAT, X_tr, X_val):
    # store y values for test and train to distill
    if LABEL_FORMAT == 'normal':  # create the distilled versions for new normal predictions
        save_distilled_y(model, X_tr, "np_arrays/y_tr_distilled")
        save_distilled_y(model, X_val, "np_arrays/y_val_distilled")
        model.save_weights("saved_models/covid-19-model-normal.h5")
    if LABEL_FORMAT == 'distill':
        model.save_weights("saved_models/covid-19-model-distilled.h5")
    if LABEL_FORMAT == 'smooth':
        model.save_weights("saved_models/covid-19-model-smoothed.h5")

def load_weights(model, LABEL_FORMAT):
    if LABEL_FORMAT == 'normal':
        model.load_weights("saved_models/covid-19-model-normal.h5")
    if LABEL_FORMAT == 'distill':
        model.load_weights("saved_models/covid-19-model-distilled.h5")
    if LABEL_FORMAT == 'smooth':
        model.load_weights("saved_models/covid-19-model-smoothed.h5")


import datetime

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model, X_tr, y_tr, X_val, y_val, X_te, y_te, SAVE_BEST, current_dt):
        self.model = model
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.X_te = X_te
        self.y_te = y_te
        self.myHistory = np.array([0, 0, 0])
        self.best_accuracy = 0
        self.SAVE_BEST = SAVE_BEST
        plt.ion()
        plt.show()
        self.best_model_name = "saved_models/covid-19-model-current-run-best" + current_dt + ".h5"

    def on_epoch_end(self, epoch, logs=None):
        train_accuracy = 100*self.model.evaluate(self.X_tr, self.y_tr, verbose=0)[1]
        validation_accuracy = 100 * self.model.evaluate(self.X_val, self.y_val, verbose=0)[1]
        test_accuracy = 100*self.model.evaluate(self.X_te, self.y_te, verbose=0)[1]

        self.myHistory = np.vstack((self.myHistory, [train_accuracy, validation_accuracy, test_accuracy]))

        ps = "Train: " + str(train_accuracy) + "\nValidation: " + str(validation_accuracy) + "\nTest: " + str(test_accuracy)
        print(ps)

        x_vals = np.arange(np.shape(self.myHistory)[0])
        train_line = self.myHistory.T[0]
        val_line = self.myHistory.T[1]
        test_line = self.myHistory.T[2]

        plt.clf()
        plt.plot(x_vals, train_line, label="train acc")
        plt.plot(x_vals, val_line, label="validation acc")
        plt.plot(x_vals, test_line, label="test acc")

        title = "Accuracy after " + str(epoch+1) + " epochs"
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(x_vals)
        plt.grid()
        plt.title(title)
        plt.legend()
        plt.draw()
        plt.pause(1)

        if test_accuracy > self.best_accuracy and self.SAVE_BEST:
            self.best_accuracy = test_accuracy
            self.model.save_weights(self.best_model_name)
