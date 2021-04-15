from sklearn.metrics import log_loss

def evaluate_scores(y_predict, y_prob, y_true):
    """
        This function will return a dictionary of evaluation scores from model predictions and true values. 
    """

    return log_loss(y_prob, y_true)