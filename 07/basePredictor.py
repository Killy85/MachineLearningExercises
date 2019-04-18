from sklearn.model_selection import train_test_split
import timeit
import functools

class BasePredictor:

    def __init__(self,input, target, test_size=0.2):
        """Create a predictor class, splitting data and setting class number

        Arguments:
            input {array} -- Element we want to train on
            target {array} -- Corresponding value of the element

        Keyword Arguments:
            test_size {float} -- Size of the test sample (default: {0.2})
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            input,target,
            test_size=test_size)
        self.test_values = [(self.x_test[index], value)
                            for index, value in enumerate(self.y_test)]
        self.classes = set(target)


    def _generate_classifier(self):
        """This method is used to generate the classifier for the predictor
        Raises:
            NotImplementedError: Implemented on child class
        """
        raise NotImplementedError("This method is not implemented")

    def _predict(self, classifiers):
        """This method is used to run tests with the test set on
        the generated classifiers

        Arguments:
            classifiers {sklear.Model} -- The model(s) we generated
            and we wan't to test

        Raises:
            NotImplementedError: Implemented on child class
        """
        raise NotImplementedError("This method is not implemented")


    def run_predict(self):
        """Train classifier for data stored and test it
        
        Returns:
            Dict -- Stats about the execution
        """
        time_train = timeit.timeit(self._generate_classifier, number=10)
        predict = self._generate_classifier()
        timer_exec = timeit.Timer(functools.partial(self._predict,predict))
        timer_exec = timer_exec.timeit(10)
        return self._predict(predict), time_train, timer_exec




