from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, log_loss, plot_confusion_matrix
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import matplotlib.pyplot as plt

class GBM:
    def __init__(self, sift_feature_file):
        """ 
            Takes path to a GBM dataset file.
        """
        self.model = LGBMClassifier(objective="multiclass")
        self.load_data(sift_feature_file)

    def load_data(self, sift_feature_file):
        """
            Assign training data and labels from a .csv of extracted features.
        """
        feature_data = pd.read_csv(sift_feature_file)
        self.train_data = feature_data.iloc[:, 2:257]
        self.train_labels = feature_data["label_name"]
        self.unique_labels = np.unique(feature_data["label_name"])

    def plot_confusion_matrix(self):
        """
            Perform a single train/test split and plot the confusion matrix for all labels.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.train_labels, stratify=self.train_labels)
        self.model.fit(X_train, y_train)
        fig, ax = plt.subplots()
        fig.set_size_inches((10,10))
        plot_confusion_matrix(self.model, X_test, y_test, ax=ax)
        plt.show()

    def cross_validate(self, folds): 
        """ 
            Run SciKit cross validation for the GBM model.
            Return different metrics, including accuracy and log loss.
        """
        log_loss_scorer = make_scorer(log_loss, labels=np.unique(self.train_labels), needs_proba=True)
        accuracy_scorer = make_scorer(accuracy_score)
        scoring = {
            'accuracy': accuracy_scorer,
            'log_loss': log_loss_scorer,
        }
        return cross_validate(self.model, self.train_data, self.train_labels, cv=folds, scoring=scoring)


    
