from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
def get_metrics_by_epoch(y_pred_epochs, y_true_epochs):

    # Get number of labels and number of epochjs
    num_labels = np.unique(y_pred_epochs).shape[0]
    num_epochs = len(y_pred_epochs)

    # Initialize the metric dictionary
    metric_dict = {
        "epoch_precisions": [],
        "epoch_recalls": [],
        "epoch_f1": []
    }

    # Fill up the metric dictionary.
    # Each metric is a 2d array,
    # where the first dimension is the label
    # and the second is the epoch.
    # Each value represents a metric for the label at that epoch.
    for label_index in range(num_labels):
        metric_dict["epoch_precisions"].append([])
        metric_dict["epoch_recalls"].append([])
        metric_dict["epoch_f1"].append([])

        for epoch_index in range(num_epochs):
            y_pred = y_pred_epochs[epoch_index]
            y_true = y_true_epochs[epoch_index]
            epoch_confusion_matrix = multilabel_confusion_matrix(y_pred, y_true)

            # Gather true positives, false positives, false negatives, true negatives
            # (tp, fp, fn, tn)
            tp = epoch_confusion_matrix[label_index][1][1]
            fp = epoch_confusion_matrix[label_index][0][1]
            fn = epoch_confusion_matrix[label_index][1][0]
            tn = epoch_confusion_matrix[label_index][0][0]
        
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)

            # Use them to derive precisions, recalls, accuracies per class
            f1 = 2 * (precision * recall) / (precision + recall)
            metric_dict["epoch_precisions"][label_index].append(precision)
            metric_dict["epoch_recalls"][label_index].append(recall)
            metric_dict["epoch_f1"][label_index].append(f1)
    return metric_dict