import timeit
from sklearn import datasets
from logistic_regression_ovo import OvOPredictor
from logistic_regression_ovr import OvRPredictor
from random_forest import RandomForestPredictor
from svm_classifier import SVMPredictor


#Awfull Hack to mute warning about 
# convergence issues
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def benchmark_classifiers(dataset):
    data = dataset['data']
    target  = dataset['target']
    predictors = {
                'One Versus Rest - Using Logistic Regression' : OvRPredictor, 
                'One Versus One - Using Logistic Regression' : OvOPredictor,
                'Random Forest Predictor': RandomForestPredictor,
                'Support Vector Machine': SVMPredictor }

    for key in predictors.keys():
        predictor = predictors[key](data, target)
        stats, time_train, time_exec = predictor.run_predict()
        correctness = stats['correctness']
        precision = stats['precision']
        recall =  stats['recall']
        f_measure = stats['f1']
        print(f' Classifier {key} \n ------------------\n')
        print(f' Correctness : {correctness:.2f}% \n Precision : {precision:.2f}% \n Recall : {recall:.2f}% \n F1 Measure : {f_measure:.2f}%')
        print(f' Whole execution process lasted {time_train+time_exec:.2f} seconds (mean of 10 executions)\n')
        print(f' Training length : {time_train:.2f} seconds (mean of 10 executions)\n')
        print(f' Testing length : {time_exec:.2f} seconds (mean of 10 executions)\n')


if __name__ == '__main__':
    digits = datasets.load_digits()
    print('Benchmarking differents classifiers for the digit dataset')
    benchmark_classifiers(digits)

    california_housing = datasets.fetch_olivetti_faces(shuffle=True, random_state=25)
    print('Benchmarking differents classifiers for the olivetti faces dataset')
    benchmark_classifiers(california_housing)
    