import numpy as np
from random import randint

class Neuron:

    def __init__(self, dimension):
        # Instantiate the neuron memory with random data (between -1 and 1)
        self.data = np.asarray([randint(0,1) for elem in range(0, dimension)])

    def learn(self, input_data):
        """This method is used to update neuron memory while training

        Arguments:
            input_data {np.array} -- The state of the neuron we train from
        """ 
        self.data += ((input_data -
                    np.min(input_data))/np.ptp(input_data)).astype(int)

    def forget(self, input_data):
        """This method is used to update neuron memory while training

        Arguments:
            input_data {np.array} -- The state of the neuron we train from
        """
        self.data -= ((input_data -
                    np.min(input_data))/np.ptp(input_data)).astype(int)
        self.data = (self.data >= 0) * self.data

    def _scale(self,):
        """Updating the memory matrix to a (0,255) matrix

        Returns:
            [array] -- Updated matrix with (0,255) values
        """
        nom = (self.data-self.data.min())/(self.data.max()-self.data.min())
        return nom*255

    def get_showable_matrix(self):
        """Return an opencv displayable matrix
        
        Returns:
            [matrix] -- An opencv displayable matrix
        """
        return self._scale()

    def get_prediction_weigth(self, comparison):
        """Compute the dot product between the memory and the input array
        
        Arguments:
            comparison {array} -- Input array to be multiplied by the memory
        
        Returns:
            [float] -- Dot product of memory and input array
        """
        return np.dot(self.data, comparison)
