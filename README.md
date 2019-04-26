# Machine Learning - Ynov 2019

This repo is the synthesis of all the exercices asked during the Machine Learning course I followed at Ynov in 2019.

The lesson was given by Jeff Abrahamson @JeffAbrahamson, whom gave us the basics needed to understand properly the base mechanism of Machine Learning


To execute the examples, ensure you have installed Python3.6 and all the libraries listed in `requirements.txt`

For python version, you may run :

```console
$ python --version
```

which should return

```console
$ python --version
Python 3.6.7
```
or higher.

We use to plot our result to make them more understandable, this means we use Tkinter. To install it, run :

```console
$ apt install python3-tk
```

you may need to run it with `sudo`

For the requirements you'll have to run 

```console
$ pip install -r requirements.txt
```
You may also look at python's [`venv`](https://docs.python.org/3/library/venv.html). That may help you to keep cleans work environnements for each of your python projects.

## Exercices details

### 01 - Statistics - pandas

During the first lesson, Jeff introduce us to [`python`](https://www.python.org/) and [`pandas`](https://pandas.pydata.org/). 

The first one is a language really helpfull while doing Machine Learning. `pandas` is a library specialized in data treatment which may be of great help when sorting and analysing data.

This folder contain a python script and a csv data file. The script import data from the file and outputs a basic analysis of what we can see in it.

To run it, imagining you have a shell at the root of this git project, just type:

```console
$ cd 01
$ python pandas_test.py
```

This should output basics data (mean, ...) and show graphics corresponding to the data imported.

### 02 - Linear Regression

He then introduces us to [`linear regression`](https://en.wikipedia.org/wiki/Linear_regression) which is a way for us to estimate a model as a linear function.

To do so we have to use the `gradient descent` algorithm which is aimed at finding a local minimal value.

Using this on our cost function, this will help to approximate the best values of θ0 and θ1 to minimize the cost function value.

So we did implement the gradient descent in 'gradient_descent.py' and use the one available in scikit learn in `gradient_descent_w_scikit.py`.

Those return the linear model we calculated before and a normalized respresentation of error there is between the points use to calculate and the actual model.

You can see the work that has been made using the following command:

```console
$ cd 02
$ python gradient_descent.py
```

for the home-made version, and

```console
$ cd 02
$ python gradient_descent_w_scikit.py
```

for the scikit version.

### 03 - Logistic Regression

In order to explain classification problem, we then discover how to create `Logistic Regression` models helping us to classify elements accordingly to train set.

This exercice uses the data we retrieve from the `sklearn` library which give use matrix corresponding to 8x8 handwritten digits.

So we explored two possibilities:
* One versus One - We compare values one versus one for each element we assume to classify.
    
    For example, the data we use declare 10 possible outputs, so we have to create all combinations of pairs we want to compare.

    So we will have to create 45 classifiers and use them to predict the values of elements in test data.

    This part of the exercice is described in the following functions `generateOvOClassifier` and `predictOvO`. They also print stats about the models.

* One versus Rest - We compare values one versus rest and the classifiers tells us if the test data is from the class we want to recognize or not

    Here we only create 10 classifiers, one for each digits, and we pass test data into each of those. The prediction and the probability of thoses predictions are used to determine which possibility is the most likely to be true.

    This part of the exercice is described in the following functions `generateOvRClassifier` and `predictOvR`. They also print stats about the models.

To run it, imagining you have a shell at the root of this git project, just type:

```console
$ cd 03
$ python logistic_regression.py
```


### 04 - Infinitesimal calculus

We were introduced to infinitesimal calculus and had to do some calculation using python.


Two things to do:
* Approach the value **e** using the fact that if **a == e** then **ln(a) = 1**
    That's what the `approching_e.py` script do.
    To launch it you just have to type the following in your shell:
    
```console
$ cd 04
$ python approching_e.py
```

* Calculate the value of **e** locally using the Taylor's sequence

    That's what the `taylor_for_x.py` script do.
    To launch it you just have to type the following in your shell:
    
```console
$ cd 04
$ python approching_e.py
```

    you can also choose the **x** you want to calculate using:

```console
$ cd 04
$ python approching_e.py 15
```
This will calculate the value around 15.

### 05 - Recommendation

We then studied recommendation algorithms. Thoses algorithms aims a predicting content for users according to differents values.

There is 3 types of recommendations:

* Content-based recommendation
* Collaborative recommendation
* Knowledge-Based recommendation

We studied the first 2 of them, the third one being expensive and hard to apply.

The first script, `movielens.py`, is downloading the movielens corpus which contain matching between users and film and the corresponding rating.

We train a model, using [`surprise`](http://surpriselib.com/) which ,thanks to linear regression, is able to predict how a user will rate a film according to how he noted other ones, and the way other users scored them too. At first launch, you may have to download the corpus!

To launch it, type the following:

```console
$ cd 05
$ python movielens.py
```

The second one aim at creating a recommendation engine enabling us to choose **n** papers related to the one we choose. This is the system we may use if we manage a website and we wand to offer more papers to our users to read.

At the moment it takes scientific papers from this [repo](https://github.com/elsevierlabs/OA-STM-Corpus) to train and test the model.

All the documents are available directly in this repo. What the programme is showing are the most relevant elements accordingly to the test element.

To run this example, you just have to type:

```console
$ cd 05
$ python tfidf.py
```

### 07 - Benchmarking differents techniques

During this TP the goal was to compare the differents techniques for the same use case.
To do so, I updated the classifiers I already use to have the same base class.

You may find all of them to derive from the `basePredictor.py` file.

In the main file, I run each classifier, showing the results in terms of precision but also efficiency and speed.

To run the program, you may do the following:

```console
$ cd 07
$ python main.py
```

It will output stats for each classifier I already implemented

#### *Here come a new classifier*

According to TP 8, we had to create a *handmade* neural network, which must be able to recognize digits. It has been added to the benchmark but is only available for the MNIST dataset.


### 08 - Our first neural network

It was time for us to be introduced to the neural network theory.
To be sure we understand the theory, we were asked to create a program doing a *perceptron*-like algorithm using the underlying scheme of a neural network.

You can find information on what has been made with commentary on the code.

To launch the example, just run :

```console
$ cd 08
$ python main.py
```

You may activate/deactivate the display of the memory state of the neuron by changed the state of `show_weigth` in `main.py l.24`.