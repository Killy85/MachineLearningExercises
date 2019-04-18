import numpy as np
import operator

from basePredictor import BasePredictor
from sklearn.linear_model import LogisticRegression


class OvRPredictor(BasePredictor):

    def _generate_classifier(self):
        o_vs_r_classifiers = {}
        for elem in self.classes:
            class_valid = [self.x_train[index] for index, value in enumerate(self.y_train) if value == elem]
            class_invalid = [self.x_train[index] for index, value in enumerate(self.y_train) if value != elem]
            value = [1] * len(class_valid) + [0] * len(class_invalid)
            learn = class_valid + class_invalid
            o_vs_r_classifiers["%d_rest" % elem] = LogisticRegression(multi_class='ovr', solver='lbfgs').fit(learn, value)
        return o_vs_r_classifiers


    def _predict(self, classifiers):
        results = {}
        stats = {'TP' :0, 'FP' : 0, 'TN': 0, 'FN': 0}
        i=0
        for elem in self.test_values:
            intern_result = {}
            for name, classifier in classifiers.items():
                result = classifier.predict([elem[0]])
                result_proba = classifier.predict_proba([elem[0]])
                intern_result[name.split('_')[0]] = result_proba[0][1]
                if result == 0:
                    if int(name.split('_')[0]) != elem[1]:
                        stats['TN'] += 1
                    else:
                        stats['FN'] += 1
                else:
                    if int(name.split('_')[0]) != elem[1]:
                        stats['FP'] += 1
                    else:
                        stats['TP'] += 1
            results[i] = intern_result
            i+=1
        correct = 0
        for key, elem in results.items():
            predicted = max(elem.items(), key=operator.itemgetter(1))[0]
            value = self.test_values[key][1]
            if int(predicted) == value:
                correct +=1
        
        correctness = (correct/len(results)*100)
        precision = (stats['TP']/(stats['TP'] + stats['FP'])) *100
        recall = (stats['TP']/(stats['TP'] +stats['FN']))*100
        f_measure = ((precision * recall)/(precision + recall))*2
        return {'correctness' :  correctness, 'precision' : precision,
                'recall' : recall, 'f1' : f_measure}




