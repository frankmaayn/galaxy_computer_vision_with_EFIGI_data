from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, log_loss, plot_confusion_matrix
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import matplotlib.pyplot as plt

class GBM:
    def __init__(self, sift_feature_file, num_sift_features, model = None):
        """ 
            Takes path to a GBM dataset file.
        """

        # Load a model from a file if one is supplied,
        # or create a new one if none is specified.
        if model == None:
            self.model = LGBMClassifier(objective="multiclass")
        else:
            self.model = model
        self.load_data(sift_feature_file, num_sift_features)


    def load_data(self, sift_feature_file, num_sift_features):
        """
            Assign training data and labels from a .csv of extracted features.
        """
        feature_data = pd.read_csv(sift_feature_file)
        sift_column_labels = ["SIFT_" + str(i) for i in range(num_sift_features)]
        self.train_data = np.array(feature_data[sift_column_labels])
        self.train_labels = np.array(feature_data["label_name"])
        self.unique_labels = np.unique(feature_data["label_name"])


    def train(self, test_percent, val_percent):
        # First split the data into train and test sets.
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.train_data, self.train_labels, stratify=self.train_labels, test_size=test_percent)
    
        # Then split the train data into train and validation sets.
        # TODO: Make sure validation proportion is relative to the original dataset site (I think it's splitting on the split here)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percent)

        
        self.model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose=True)



    def plot_confusion_matrix(self):
        """
            Perform a single train/test split and plot the confusion matrix for all labels.
        """
        fig, ax = plt.subplots()
        fig.set_size_inches((10,10))
        plot_confusion_matrix(self.model, self.X_test, self.y_test, ax=ax)
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


    
