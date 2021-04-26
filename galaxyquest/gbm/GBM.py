from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, log_loss, plot_confusion_matrix
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

class GBM:
    def __init__(self, sift_feature_file, label_file, num_sift_features, model = None):
        """ 
            Takes path to a GBM dataset file.
        """

        # Load a model from a file if one is supplied,
        # or create a new one if none is specified.
        if model == None:
            self.model = LGBMClassifier(objective="multiclass")
        else:
            self.model = model
        self.load_data(sift_feature_file, label_file, num_sift_features)


    def load_data(self, sift_feature_file, label_file, num_sift_features):
        """
            Assign training data and labels from a .csv of extracted features.
        """
        feature_data = pd.read_csv(sift_feature_file)
        label_data = pd.read_csv(label_file)
        feature_data = label_data.merge(feature_data.set_index("pgc_id"), on="pgc_id", how="inner")
        sift_column_labels = ["SIFT_" + str(i) for i in range(num_sift_features)]
        self.train_data = np.array(feature_data[sift_column_labels])
        self.train_labels = np.array(feature_data["category_label"])
        self.unique_labels = np.unique(feature_data["category_label"])


    def train(self, test_percent, val_percent):
        # First split the data into train and test sets.
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.train_data, self.train_labels, stratify=self.train_labels, test_size=test_percent)
    
        # Then split the train data into train and validation sets.
        # TODO: Make sure validation proportion is relative to the original dataset site (I think it's splitting on the split here)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percent)

        self.model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose=True)

    def plot_confusion_matrix(self, labels):
        """
            Perform a single train/test split and plot the confusion matrix for all labels.
        """
        fig, ax = plt.subplots()
        fig.set_size_inches((10,10))
        plot_confusion_matrix(self.model, self.X_test, self.y_test, ax=ax)
        x_locs = range(len(self.unique_labels))
        plt.xticks(x_locs, labels)
        plt.yticks(x_locs, labels)
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


    def cross_validate_predict(self, folds):
        """
            Get cross validated predictions for a few test sets.
        """
        cv_truth_labels = []
        cv_predictions = []
        kf = StratifiedKFold(n_splits = folds)
        for train_index, test_index in kf.split(np.zeros(len(self.train_labels)), self.train_labels):
            X_train, X_test = self.train_data[train_index], self.train_data[test_index]
            y_train, y_test = self.train_labels[train_index], self.train_labels[test_index]

            self.model.fit(X_train, y_train)
            cv_truth_labels.append(y_test)
            cv_predictions.append(self.model.predict(X_test))
        
        return {
            "predictions": cv_predictions,
            "ground_truth_labels": cv_truth_labels
        }

        
    def tts_predict(self):
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.train_labels)
        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_test)
        return {
            "predictions": [y_predict],
            "ground_truth_labels": [y_test]
        }