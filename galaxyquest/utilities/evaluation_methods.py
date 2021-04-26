from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def get_metrics_by_epoch(y_pred_epochs, y_true_epochs, label_names = None):

    # Get number of labels and number of epochjs
    num_labels = len(np.unique(y_true_epochs[0]))
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
            epoch_confusion_matrix = multilabel_confusion_matrix(y_pred, y_true, labels=label_names)

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

def plot_metric_byclass(epoch_metrics, metric, labels, title):

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(title)
    ax.set_title("Final " + metric + " by class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")

    metric_list = []

    for i in range(len(epoch_metrics["epoch_" + metric])):
        metric_list.append(epoch_metrics["epoch_" + metric][i][-1])

    x_locs = range(len(metric_list))
    for i,  v in enumerate(metric_list):
        plt.text(x_locs[i] - 0.12, v + 0.01, "{:.2f}".format(v))

    
    plt.bar(x_locs, metric_list)
    plt.xticks(x_locs, labels)
