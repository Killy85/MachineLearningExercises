import timeit
from sklearn import datasets
from logistic_regression_OvO import OvOPredictor
from logistic_regression_OvR import OvRPredictor
from randomForest import RandomForestPredictor


#Awfull Hack to mute warning about 
# convergence issues
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():
    digits = datasets.load_digits()
    data = digits['data']
    target  = digits['target']
    predictors = {
                'One Versus Rest - Using Logistic Regression' : OvRPredictor, 
                'One Versus One - Using Logistic Regression' : OvOPredictor,
                'Random Forest Predictor': RandomForestPredictor }

    for key in predictors.keys():
        predictor = predictors[key](data, target)
        time = timeit.timeit(predictor.run_predict, number=10)
        stats = predictor.run_predict()
        correctness = stats['correctness']
        precision = stats['precision']
        recall =  stats['recall']
        f_measure = stats['f1']
        print(f' Classifier {key} \n ------------------\n')
        print(f' Correctness : {correctness:.2f}% \n Precision : {precision:.2f}% \n Recall : {recall:.2f}% \n F1 Measure : {f_measure:.2f}%')
        print(f' Whole execution process lasted {time:.2f} seconds (mean of 10 executions)\n')


if __name__ == '__main__':
    main()
    