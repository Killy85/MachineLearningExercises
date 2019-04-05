import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import itertools
import operator


class RandomForestPredictor:

    def __init__(self,input, target, test_size=0.2):
        self.x_train,self.x_test, self.y_train,  self.y_test = train_test_split(input,target, test_size=test_size, random_state = 21)
        self.test_values = [(self.x_test[index],value) for index, value in enumerate(self.y_test)]
        self.classes = set(target)


    def _createForest(self):
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        return forest.fit(self.x_train, self.y_train)

    def _testForest(self, forest):
        stats = {'TP' :0, 'FP' : 0, 'TN': 0, 'FN': 0}
        for elem in self.test_values:
            predicted = forest.predict([elem[0]])
            if predicted[0] == elem[1]:
                stats['TP'] +=1
            else:
                stats['FP'] +=1
        correctness = (stats['TP']/len(self.test_values)*100)
        precision = (stats['TP']/(stats['TP'] + stats['FP'])) *100
        recall = (stats['TP']/(stats['TP'] +stats['FN']))*100
        f_measure = ((precision * recall)/(precision + recall))*2
        return {'correctness' :  correctness, 'precision' : precision,
                'recall' : recall, 'f1' : f_measure}

        

    def run_predict(self):
        forest = self._createForest()
        return self._testForest(forest)




