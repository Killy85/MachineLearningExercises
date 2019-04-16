import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
import operator

#Awfull Hack to mute warning about 
# convergence issues
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def generateOvRClassifier(classes):
    """ This method is used to generate One versus Rest Logistic regression
    model for all the elements of the classes passed as parameters.

        For each OvR classifier we split data as following
        One array containing all the data from our current class
        One array containing the rest
    """
    o_vs_r_classifiers = {}
    for elem in classes:
        class_valid = [x_train[index] for index, value in enumerate(y_train) if value == elem]
        class_invalid = [x_train[index] for index, value in enumerate(y_train) if value != elem]
        value = [1] * len(class_valid) + [0] * len(class_invalid)
        learn = class_valid + class_invalid
        # We create and store a Logistic regression that fit our data
        o_vs_r_classifiers["%d_rest" % elem] = LogisticRegression(multi_class='ovr',solver='lbfgs').fit(learn, value)
    return o_vs_r_classifiers


def predictOVR(test_values, o_vs_r_classifiers):
    """We Compare the results given by our classifiers
    with test datas, to benchmark the efficiency of the solution
    """
    results = {}
    stats = {'TP' :0, 'FP' : 0, 'TN': 0, 'FN': 0}
    i=0
    for elem in test_values:
        intern_result = {}
        for name, classifier in o_vs_r_classifiers.items():
            result = classifier.predict([elem[0]])
            result_proba = classifier.predict_proba([elem[0]])
            intern_result[name.split('_')[0]] = result_proba[0][1]
            if result == 0:
                if int(name.split('_')[0])!= elem[1]:
                    # The classifiers answered 'False' and it was False
                    stats['TN'] += 1
                else:
                    # The classifiers answered 'False' and it was True
                    stats['FN'] +=1
            else:
                if int(name.split('_')[0])!= elem[1]:
                    # The classifiers answered 'True' and it was False
                    stats['FP'] += 1
                else:
                    # The classifiers answered 'True' and it was True
                    stats['TP'] +=1
        results[i] = intern_result
        i+=1
    correct = 0
    for key, elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct +=1
    # Calculating perfomance indicator to compare Solutions
    prct = (correct/len(results)*100)
    precision = (stats['TP']/(stats['TP'] + stats['FP'])) *100
    recall = (stats['TP']/(stats['TP'] +stats['FN']))*100
    f_measure = ((precision * recall)/(precision + recall))*2
    print(f"""Following Stats are for One vs Rest \n  Score for the program : 
    Precision {precision:.2f}% \n    Recall {recall:.2f}% \n    F1 measure {f_measure:.2f}%\n    Correctness {prct:.2f}%""")

def generateOvOClassifier(classes):
    """ This method is used to generate One versus One Logistic regression
    model for all the elements of the classes passed as parameters.

        For each OvO classifier we split data as following
        One array containing all the data from one of the current class
        One array containing the data for the other class
    """
    o_vs_o_classifiers = {}
    # The following line generate all combinations possible 
    # of size 2 for the given list
    for elem in itertools.combinations(classes,2):
        class0 = [x_train[index] for index, value in enumerate(y_train) if value == elem[0]]
        class1 = [x_train[index] for index, value in enumerate(y_train) if value == elem[1]]
        value = [0] * len(class0) + [1] * len(class1)
        learn = class0 + class1
        o_vs_o_classifiers['%d_%d'%elem] = LogisticRegression(solver='lbfgs').fit(learn, value)
    return o_vs_o_classifiers

def predictOVO(test_values, o_vs_o_classifiers):
    """We Compare the results given by our classifiers
    with test datas, to benchmark the efficiency of the solution
    """
    results = {}
    stats = {'TP' :0, 'FP' : 0, 'TN': 0, 'FN': 0}
    i=0
    for elem in test_values:
        intern_result = {}
        for name,classifiers in o_vs_o_classifiers.items():
            result = classifiers.predict([elem[0]])
            members = name.split('_')
            if intern_result.get(members[result[0]]):
                intern_result[members[result[0]]] += 1
            else:
                intern_result[members[result[0]]] = 1
            # Here the stats are a bit tricky to calculate
            if str(elem[1]) in members:
                # We check the result is in the members that were compared
                if int(members[result[0]])== elem[1]:
                    # If the predicted member is the result, that's a True positive
                    stats['TP'] += 1
                else:
                    # Else a false positive
                    stats['FP'] +=1
            else:
                # If the result is out, we just can say it's a True negative
                stats['TN'] +=1
        results[i] = intern_result
        i+=1
    correct = 0
    for key,elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct += 1
    # Calculating perfomance indicator to compare Solutions
    prct = (correct/len(results)*100)
    precision = (stats['TP']/(stats['TP'] + stats['FP'])) *100
    recall = (stats['TP']/(stats['TP'] +stats['FN']))*100
    f_measure = ((precision * recall)/(precision + recall))*2
    print(f"""Following Stats are for One vs One \n  Score for the program :
    Precision {precision:.2f}% \n    Recall {recall:.2f}% \n    F1 measure {f_measure:.2f}% \n    Correctness {prct:.2f}%""")

if __name__ == "__main__":
    digits = datasets.load_digits()
    data = digits['data']
    target  = digits['target']
    classes = set(target)
    # Splitting the data to get train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    # We generate an array containing tuples (images,value)
    test_values = [(x_test[index],value) for index, value in enumerate(y_test)]
    # Create the O v O classifiers
    o_vs_o_classifiers = generateOvOClassifier(classes)
    # Launch the loop to predict elem in test values 
    predictOVO(test_values, o_vs_o_classifiers)
    # We generate the OvR classifiers
    ovrclassifier = generateOvRClassifier(classes)
    # And use it to test on our values
    predictOVR(test_values,ovrclassifier)
