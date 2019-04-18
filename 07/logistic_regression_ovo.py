import operator
import itertools

from basePredictor import BasePredictor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



class OvOPredictor(BasePredictor):

    def _generate_classifier(self):
        o_vs_o_classifiers = {}
        for elem in itertools.combinations(self.classes,2):
            class0 = [self.x_train[index] for index, value in enumerate(self.y_train) if value == elem[0]]
            class1 = [self.x_train[index] for index, value in enumerate(self.y_train) if value == elem[1]]
            value = [0] * len(class0) + [1] * len(class1)
            learn = class0 + class1
            o_vs_o_classifiers['%d_%d'%elem] = LogisticRegression(solver='lbfgs').fit(learn, value)
        return o_vs_o_classifiers

    def _predict(self, classifiers):
        """
        TO DO : STATS
        """
        results = {}
        stats = {'TP' :0, 'FP' : 0, 'TN': 0, 'FN': 0}
        i=0
        for elem in self.test_values:
            intern_result = {}
            for name,classifier in classifiers.items():
                result = classifier.predict([elem[0]])
                members = name.split('_')
                if intern_result.get(members[result[0]]):
                    intern_result[members[result[0]]] += 1
                else:
                    intern_result[members[result[0]]] = 1
                if str(elem[1]) in members:
                    if int(members[result[0]])== elem[1]:
                        stats['TP'] += 1
                    else:
                        stats['FP'] +=1
                else:
                    stats['TN'] +=1
            results[i] = intern_result
            i+=1
        correct = 0
        for key,elem in results.items():
            predicted = max(elem.items(), key=operator.itemgetter(1))[0]
            value = self.test_values[key][1]
            if int(predicted) == value:
                correct += 1
        correctness = (correct/len(results)*100)
        precision = (stats['TP']/(stats['TP'] + stats['FP'])) *100
        recall = (stats['TP']/(stats['TP'] +stats['FN']))*100
        f_measure = ((precision * recall)/(precision + recall))*2
        return {'correctness' :  correctness, 'precision' : precision,
                'recall' : recall, 'f1' : f_measure}




