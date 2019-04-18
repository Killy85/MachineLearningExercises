from basePredictor import BasePredictor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



class SVMPredictor(BasePredictor):

    def _generate_classifier(self):
        forest = SVC(gamma='scale', decision_function_shape='ovo')
        return forest.fit(self.x_train, self.y_train)

    def _predict(self, classifiers):
        stats = {'TP' :0, 'FP' : 0, 'TN': 0, 'FN': 0}
        for elem in self.test_values:
            predicted = classifiers.predict([elem[0]])
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





