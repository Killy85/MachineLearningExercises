import numpy as np
import cv2
from PIL import Image
import time

from neuron import Neuron

def border(img):
    bordersize = 10
    return cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize,
                        right=bordersize, borderType=cv2.BORDER_CONSTANT, value=0)


class Network:
    """The class is a basic implementation of a neural network.
        It accept a little parametrisation, but some of the method
        Make it hard to do something else than treating 8x8 images
        containing data between 0 and 255 (grayscale)
    """

    def __init__(self, neuron_number, dimension):
        """Instatiate the network
        
        Arguments:
            neuron_number {Integer} -- The number of neurons
                                                    we have to instantiate
            dimension {Integer} -- The dimension of the neurones we want to use
        """
        self.neurons = [Neuron(dimension) for elem in range(neuron_number)]

    def fit(self, X_data, Y_data, show_weigth=False):
        """Fit the data into the network
        
        Arguments:
            X_data {array} -- The training data we want to fit
            Y_data {array} -- The classes correspondance for those data
        
        Keyword Arguments:
            show_weigth {bool} -- Toggle display of neurons memory
                                                        (default: {False})
        """
        for index, image in enumerate(X_data):
            img = None
            scores = [elem.get_prediction_weigth(image)
                      for elem in self.neurons]
            predicted = scores.index(max(scores))
            if predicted != int(Y_data[index]):
                self.neurons[int(predicted)].forget((image > 13) * image)
                self.neurons[int(Y_data[index])].learn((image > 13) * image)

            if show_weigth:
                img = self.showSprite([elem.get_showable_matrix()
                                 for elem in self.neurons])
        cv2.destroyAllWindows()
        return img


    def showSprite(self, spritesData):
        """Display the neurons memory using opencv

        Arguments:
            spritesData {array} -- The memory state we want to display
        """
        images = [(255-elem) for elem in spritesData]
        images = [cv2.resize(frame.reshape(8, 8), (256, 256),
                  interpolation=cv2.INTER_AREA)
                  for frame in images]

        vertical_1 = np.vstack((border(images[0]), border(images[5])))
        vertical_2 = np.vstack((border(images[1]), border(images[6])))
        vertical_3 = np.vstack((border(images[2]), border(images[7])))
        vertical_4 = np.vstack((border(images[3]), border(images[8])))
        vertical_5 = np.vstack((border(images[4]), border(images[9])))

        img = np.hstack((vertical_1, vertical_2, vertical_3,
                        vertical_4, vertical_5))
        cv2.imshow("Window", np.uint8(img))
        cv2.waitKey(1)
        return np.uint8(img)

    def test(self, x_test, y_test):
        """Testing the neural net according to test data

        Arguments:
            x_test {array} -- Input data we wan't to test
            y_test {array} -- Correspondind value

        Returns:
            tuple -- the ratio of good prediction and 
            the number of item rightly predicted
        """
        correct = 0
        for index, test in enumerate(x_test):
            scores = [elem.get_prediction_weigth(test)
                      for elem in self.neurons]
            predicted = scores.index(max(scores))
            if predicted == int(y_test[index]):
                correct += 1
        return ((correct/len(x_test)), correct)
