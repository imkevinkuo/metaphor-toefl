import math

import numpy as np
import torch
from sklearn import metrics
from torch.autograd import Variable


def evaluate(evaluation_dataloader, model, criterion, using_GPU):
    """
    Evaluate the model on the given evaluation_dataloader

    :param threshold_prob:
    :param evaluation_dataloader:
    :param model:
    :param criterion: loss criterion
    :param using_GPU: a boolean
    :return:
     average_eval_loss
     2x2 confusion matrix, not separated by pos tag
    """
    # Set model to eval mode, which turns off dropout.
    model.eval()

    # total_examples = total number of words
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((2, 2))
    for (eval_text, eval_lengths, eval_labels, eval_features) in evaluation_dataloader:
        eval_text = Variable(eval_text)
        eval_lengths = Variable(eval_lengths)
        eval_labels = Variable(eval_labels)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()

        # predicted shape: (batch_size, seq_len, 2)
        predicted = model(eval_text, eval_lengths)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
        total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
        # get 0 or 1 predictions
        _, predicted_labels = torch.max(predicted.data, 2)
        total_examples += eval_lengths.size(0)
        confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, eval_labels.data)

    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()

    # Set the model back to train mode, which activates dropout again.
    model.train()
    return average_eval_loss.item(), print_info(confusion_matrix)


def update_confusion_matrix(matrix, predictions, labels):
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            matrix[predictions[i][j]][labels[i][j]] += 1
    return matrix


def print_info(matrix):
    precision = 100 * matrix[1, 1] / np.sum(matrix[1])
    recall = 100 * matrix[1, 1] / np.sum(matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (matrix[1, 1] + matrix[0, 0]) / np.sum(matrix)
    return np.array([accuracy, precision, recall, f1])


def get_batch_predictions(predictions, pos_seqs):
    """
    :param predictions: a numpy array of shape (batch_size, max_seq_len)
    :param pos_seqs: a list of variable-length indexed pos sequence
    :return: a list of variable-length predictions. each inner list is prediction for a sentence
    """
    pred_lst = []
    for i in range(len(pos_seqs)):  # each example i.e. each row
        indexed_pos_sequence = pos_seqs[i]
        prediction_padded = predictions[i]
        cur_pred_lst = []
        for j in range(len(indexed_pos_sequence)):  # inside each example: up to sentence length
            cur_pred_lst.append(prediction_padded[j])
        pred_lst.append(cur_pred_lst)
    return pred_lst


def get_optimal_threshold_f1(val_y, prob_y):
    precision, recall, thresholds = metrics.precision_recall_curve(val_y, prob_y, pos_label=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    thresh_optim_f1 = thresholds[np.where(f1 == max(f1))[0].item()]
    return thresh_optim_f1


def get_optimal_threshold_tpfp(val_y, prob_y):
    # Raw count of TP - FP
    precision, recall, thresholds = metrics.precision_recall_curve(val_y, prob_y, pos_label=1)
    thresh_optim = 0
    max_diff = 0
    for threshold in thresholds:
        try:
            pos_pred_inds = np.concatenate(np.argwhere(prob_y > threshold))
        except ValueError:
            # Nothing is above threshold, so make empty array
            # print(np.argwhere(prob_y > threshold))
            pos_pred_inds = []
        tp = sum(val_y[pos_pred_inds])
        fp = len(pos_pred_inds) - tp
        if tp - fp > max_diff:
            max_diff = tp - fp
            thresh_optim = threshold
    print(f"p={threshold} gives {max_diff} extra swaps")
    return thresh_optim
