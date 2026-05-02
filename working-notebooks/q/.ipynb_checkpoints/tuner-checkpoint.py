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
    - `X`: features dataframe
    - `y`: target dataframe
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
             seed_max: int,
             test_size=0.20,
             hyperparameter: str=None,
             hyperparameter_settings=None,
             other_settings: dict={},
             metric = 'default',
             metric_min=False):
        """
        Parameters: 

        - `classifier`: classifier/regressor class
        - `seed_max`: max seed for random_state
        - `test_size`: Split size for X and y testing
        - `hyperparameter`: Optional; hyperparameter variable name of `classifier`
        - `hyperparameter_settings`: Optional; hyperparameter settings to be tested.
        - `other_settings`: Optional; other arguments for `classifier`.
        
        Sample usage:
            tuner.tune(KNeighborsClassifier, 10, 0.25, 'n_neighbors', range(1,11));
    
            C_settings = [1e-4, 1e-3,0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
            
            svc_settings = {'penalty':"l1", 'loss':'squared_hinge', 'dual':False}
            
            test_tuner.tune(LinearSVC, 10, 0.25, 'C', C_settings, svc_settings);
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
                clf = classifier(**(other_settings | {hyperparameter: param}))
                clf.fit(X_train, y_train)

                if metric=='default':
                    training_accuracy.append(clf.score(X_train, y_train))
                    test_accuracy.append(clf.score(X_test, y_test))
                else:
                    training_accuracy.append(metric(X_train, y_train, clf))
                    test_accuracy.append(metric(X_test, y_test, clf))
        
            trains[seed] = training_accuracy
            tests[seed] = test_accuracy

        if metric_min:
            best_setting = np.argmin(tests.mean(axis=1))
        else:
            best_setting = np.argmax(tests.mean(axis=1))
        
        id = time_ns() // 100_000_000
        self.results.loc[len(self.results)] = [id,
                                                clf_name,hyperparameter,
                                                hyperparameter_settings[best_setting],
                                                trains.iloc[best_setting].mean(),
                                                tests.iloc[best_setting].mean()
                                               ]
        self.hyperparams[id] = hyperparameter_settings
        self.trains[id] = trains
        self.tests[id] = tests
        pass
        
    def plot_hyperparameter_tuning(self,
                                   tune_id,
                                   logscale: bool=False,
                                   metric_name='accuracy'):

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
                     yerr=trains.std(axis=1), label=f"training {metric_name}", color='blue', marker='o', linestyle='dashed', markersize=15)
        plt.errorbar(hyperparameter, tests.mean(axis=1),
                     yerr=tests.std(axis=1), label=f"test {metric_name}", color='red', marker='^', linestyle='-', markersize=15)
        plt.ylabel(f"{metric_name}", fontsize=15)
        plt.xlabel(hparam_name,fontsize=15)
        plt.legend()
        pass
        