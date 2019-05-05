from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
from basePredictor import BasePredictor
tf.logging.set_verbosity(tf.logging.ERROR)


class MLPMNISTPredictor(BasePredictor):

    def __init__(self, input, target, test_size=0.2):
        super().__init__(input, target)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        self.batch_size = 128
        self.num_classes = len(self.classes)
        self.epochs = 20

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        self.model = Sequential()
        self.history = None

    def _generate_classifier(self):
        self.model.add(Dense(512, activation='relu', input_shape=(self.x_train.shape[1],)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=RMSprop(),
                           metrics=['accuracy'])
        self.history = self.model.fit(self.x_train, self.y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=0,
                                      validation_data=(self.x_test, self.y_test))

    def _predict(self, classifiers):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return {'correctness': score[0]*100, 'precision': 0,
                'recall': 0, 'f1': 0}