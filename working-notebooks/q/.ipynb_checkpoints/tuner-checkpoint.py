import pandas as pd
import numpy as np
import pylab as plot
import matplotlib.pyplot as plt
from time import time_ns
from sklearn.model_selection import train_test_split


class Tuner():
    """
    Wrapper/UI that helps in testing models over the same data.

    Currently only works for single hyperparameter classifiers/regressors.

    Results of past tunings are stored in `self.results`
    
    Params:
        `X`: features dataframe
        `y`: target dataframe
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = pd.DataFrame(columns=['tune_id',
                                             'classifier',
                                             'hyperparameter',
                                             'best_hyperparameter_setting',
                                             'test_score',
                                             'train_score'])
        self.trains = {}
        self.tests = {}
        self.hyperparams = {}
        pass

    def tune(self,
             classifier,
             seed_max,
             test_size=0.20,
             hyperparameter=None,
             hyperparameter_settings=None,
             other_settings: dict={}):
        """

        Sample usage:
        
        """
        trains = pd.DataFrame()
        tests = pd.DataFrame()

        clf_name = classifier.__name__

        for seed in range(1,seed_max+1):
            X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                                self.y,
                                                                test_size=test_size,
                                                                random_state=seed)
            
            training_accuracy = []
            test_accuracy = []

            for param in hyperparameter_settings:
                clf = classifier(**(other_settings|{hyperparameter: param}))
                clf.fit(X_train, y_train)
        
                training_accuracy.append(clf.score(X_train, y_train))
                test_accuracy.append(clf.score(X_test, y_test))
        
            trains[seed] = training_accuracy
            tests[seed] = test_accuracy
        
        best_setting = np.argmax(test_accuracy)
        
        id = time_ns() // 100_000_000
        self.results.loc[len(self.results)] = [id,
                                                clf_name,hyperparameter,
                                                hyperparameter_settings[best_setting],
                                                trains[best_setting].mean(),
                                                tests[best_setting].mean()
                                               ]
        self.hyperparams[id] = hyperparameter_settings
        self.trains[id] = trains
        self.tests[id] = tests
        pass
        
    def plot_hyperparameter_tuning(self,
                                   tune_id,
                                   logscale: bool=False):

        hyperparameter = self.hyperparams[tune_id]
        trains = self.trains[tune_id]
        tests = self.tests[tune_id]
        hparam_name = self.results[self.results['tune_id']==tune_id]['hyperparameter']
        
        fig = plt.figure(figsize=(15, 6))
        if logscale:
            plt.xscale('log')
        params = {'legend.fontsize': 15, 'legend.handlelength': 2}
        plot.rcParams.update(params)
        
        plt.errorbar(hyperparameter, trains.mean(axis=1),
                     yerr=trains.std(axis=1), label="training accuracy", color='blue', marker='o', linestyle='dashed', markersize=15)
        plt.errorbar(hyperparameter, tests.mean(axis=1),
                     yerr=tests.std(axis=1), label="test accuracy", color='red', marker='^', linestyle='-', markersize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.xlabel(hparam_name,fontsize=15)
        plt.legend()
        pass
        