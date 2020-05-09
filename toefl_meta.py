import ast
import math

import h5py
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import random
import numpy as np

import data_util
import load_util_toefl
import load_util_vua
import eval_util
from model import RNNSequenceModel

using_GPU = torch.cuda.is_available()
print(f"PyTorch version: {torch.__version__}")
print(f"GPU Detected: {using_GPU}")

# These affect embedding behavior
torch.random.manual_seed(0)
torch.cuda.manual_seed_all(0)


"""
Load data and annotations
"""

# Modify this line to change the dataset used.
dataset = "TOEFL"
print(f"Loading {dataset} train/test sentences...")

raw_valid = []
if dataset == "VUA":
    raw_train, raw_test = load_util_vua.load_train_test()
elif dataset == "VT1":
    raw_train_toefl, raw_test_toefl = load_util_toefl.load_train_test()
    raw_train_vua, raw_test_vua = load_util_vua.load_train_test()
    # raw_train_vua = load_util_vua.filter_train_by_genre(raw_train_vua, ['conversation'])
    # raw_test_vua = load_util_vua.filter_test_by_genre(raw_test_vua, ['conversation'])
    # raw_train_vua = load_util_vua.filter_by_length(raw_train_vua, 4)
    # raw_test_vua = load_util_vua.filter_by_length(raw_test_vua, 4)

    raw_train, raw_test = raw_train_vua + raw_train_toefl, raw_test_vua + raw_test_toefl
elif dataset == "TOEFL":
    raw_train, raw_test = load_util_toefl.load_train_test_no_f()
elif dataset == "TOEFL_clean":
    raw_train, raw_test = load_util_toefl.load_from_pkl(dataset)
elif dataset == "TOEFL_min":
    raw_train, raw_test = load_util_toefl.load_train_test()
    raw_train = load_util_toefl.filter_raw_data(raw_train)
    raw_test = load_util_toefl.filter_raw_data(raw_test)
elif dataset == "TOEFL_AUG":
    # Might be confusing but raw_train is just training data in this case - dont split again, we already have raw_valid
    raw_train, raw_valid, raw_test = load_util_toefl.load_data_aug(dataset)
else:
    print(f"{dataset} is not a valid dataset")
    exit(0)

"""
Word Embeddings
"""
vocab = data_util.get_vocab(raw_train + raw_valid + raw_test)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = data_util.get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
print("Loading Glove embeddings...")
# glove_embeddings = None
glove_embeddings = data_util.get_glove_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
elmos_train = h5py.File(f'elmo/{dataset}_train.hdf5', 'r')
elmos_test = h5py.File(f'elmo/{dataset}_test.hdf5', 'r')
sentence_to_index_train = ast.literal_eval(elmos_train['sentence_to_index'][0])
sentence_to_index_test = ast.literal_eval(elmos_test['sentence_to_index'][0])

"""
Features and parameter setup
"""
use_glove = True
use_elmo = True

# universal_posvec, stanford_posvec. m_count, wordnet, topic_lda, cbiasup, cbiasdown, cbiasdiff
# prompt, proficiency, native_lang

other_features = []
# other_features = ['ul_vec', 'prompt_vec']
# other_features = ['ul_vec', 'stanford_posvec', 'topic_lda', 'cbiasup', 'cbiasdown', 'cbiasdiff']  # all-15
# other_features = ['ul_vec', 'wordnet', 'cbiasdiff']  # all-16
# other_features = ['ul_vec', 'wordnet', 'cbiasup', 'cbiasdown', 'cbiasdiff', 'stanford_posvec']
# other_features = ['ul_vec', 'topic_lda', 'stanford_posvec']
# other_features = ['ul_vec', 'stanford_posvec', 'wordnet', 'topic_lda', 'cbiasup', 'cbiasdown', 'cbiasdiff']  # allanno

dropouts = [0.5, 0, 0.1]
hidden_size = 300
class_weight = None
# class_weight = [1.0, 5.0]


#
glove_embeddings = glove_embeddings if use_glove else None
elmos_train = elmos_train if use_elmo else None
elmos_test = elmos_test if use_elmo else None
#


# convenience function for working with
# results = train_and_pred_lstm(*get_ith_fold_embedded(raw_train))
# print_report_by_feature('stanford_postag', *results)
def train_and_pred_lstm(fold_train, fold_valid, embedding_dim):
    fold_valid_x, fold_valid_y, fold_valid_f, fold_valid_ids = data_util.seq_to_word_level_train(fold_valid)

    # Train RNN
    train_dataloader, valid_dataloader = embedded_to_batch(fold_train, fold_valid)
    rnn_clf, optimal_score = train_model(train_dataloader, valid_dataloader, embedding_dim, verbose=False)

    # Make predictions
    id_to_pred_idx, prob_y_rnn = predict(rnn_clf, [[example[0], example[2]] for example in fold_valid])

    # Filter validation examples
    prob_y_rnn = [prob_y_rnn[id_to_pred_idx[ex_id]] for ex_id in fold_valid_ids]
    # Optimize F1
    optimal_threshold_rnn = eval_util.get_optimal_threshold_f1(fold_valid_y, prob_y_rnn)
    pred_y_rnn = np.where(prob_y_rnn > optimal_threshold_rnn, 1, 0)
    return fold_valid_f, fold_valid_y, pred_y_rnn


def get_ith_fold_embedded(train_data=raw_train, i=0, k=5):
    embedded_train, embedding_dim = embed_data(train_data, True)
    fold_train, fold_valid = data_util.get_ith_fold_raw(embedded_train, i, k)
    return fold_train, fold_valid, embedding_dim


# This ones for aug data, where validation is already separated
# fold_valid_f, fold_valid_y, pred_y_rnn = train_raw_split(fold_train, fold_valid)
def train_raw_split(train_data=raw_train, valid_data=raw_valid):
    embed_train, embed_dim = embed_data(train_data, True)
    embed_valid, embed_dim = embed_data(valid_data, True)
    model, optimal_score = train_model(*embedded_to_batch(embed_train, embed_valid), embed_dim, num_epochs=6)

    fold_valid_x, fold_valid_y, fold_valid_f, fold_valid_ids = data_util.seq_to_word_level_train(embed_valid)
    # Make predictions
    id_to_pred_idx, prob_y_rnn = predict(model, [[example[0], example[2]] for example in embed_valid])

    # Filter validation examples
    prob_y_rnn = [prob_y_rnn[id_to_pred_idx[ex_id]] for ex_id in fold_valid_ids]
    # Optimize F1
    optimal_threshold_rnn = eval_util.get_optimal_threshold_f1(fold_valid_y, prob_y_rnn)
    pred_y_rnn = np.where(prob_y_rnn > optimal_threshold_rnn, 1, 0)
    return fold_valid_f, fold_valid_y, pred_y_rnn


def embed_data(raw_data, include_label=False):
    if include_label:
        sentence_to_index = sentence_to_index_train
        elmo_embeddings = elmos_train
    else:
        sentence_to_index = sentence_to_index_test
        elmo_embeddings = elmos_test

    embedded_sentences = [data_util.embed_indexed_sequence(
        example[0], example[1], glove_embeddings, word2idx, elmo_embeddings,
        sentence_to_index[example[0]], other_features) for example in raw_data]

    if include_label:
        embedded_data = [[embedded_sentences[i],
                          [features['label'] for features in raw_data[i][1]],
                          raw_data[i][1]] for i in range(len(raw_data))]
    else:
        embedded_data = [[embedded_sentences[i], raw_data[i][1]] for i in range(len(raw_data))]

    embedding_dim = embedded_sentences[0].shape[1]
    print("Embedding dim:", embedding_dim)
    return embedded_data, embedding_dim


def embedded_to_batch(fold_train, fold_valid, batch_size=16):
    # Feed into Dataset -> DataLoader to train in batches.
    train_dataset = data_util.TextDataset([example[0] for example in fold_train],
                                          [example[1] for example in fold_train],
                                          [example[2] for example in fold_train])
    valid_dataset = data_util.TextDataset([example[0] for example in fold_valid],
                                          [example[1] for example in fold_valid],
                                          [example[2] for example in fold_valid])

    # For near-reproducible results do not use DataLoader shuffle.
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=data_util.TextDataset.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                  collate_fn=data_util.TextDataset.collate_fn)
    return train_dataloader, valid_dataloader


def train_model(train_dataloader, val_dataloader, embedding_dim, num_epochs=20, print_every=200, prefix=dataset, verbose=True):
    model = RNNSequenceModel(num_classes=2, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=1,
                             bidir=True, dropout1=dropouts[0], dropout2=dropouts[1], dropout3=dropouts[2])
    model = model.cuda() if using_GPU else model

    loss_weight = None
    if class_weight:
        loss_weight = torch.Tensor(class_weight)
        loss_weight = loss_weight.cuda() if using_GPU else loss_weight

    loss_criterion = nn.NLLLoss(weight=loss_weight, reduction='sum' if class_weight else 'mean')
    rnn_optimizer = optim.Adam(model.parameters(), lr=0.005)

    optimal_score = [0, 0, 0, 0, 0]  # Iteration, Acc, Precision, Recall, F1
    optimal_state_dict = None
    num_iter = 0  # Number of gradient updates
    train_confusion_matrix = np.zeros((2, 2))  # Keep track of training performance - resets every 200 updates
    for epoch in range(num_epochs):
        # Slower learning rate
        if epoch == num_epochs / 2:
            rnn_optimizer = optim.Adam(model.parameters(), lr=0.001)

        for (example_text, example_lengths, example_labels, example_features) in train_dataloader:
            example_text = Variable(example_text)
            example_lengths = Variable(example_lengths)
            example_labels = Variable(example_labels)
            if using_GPU:
                example_text = example_text.cuda()
                example_lengths = example_lengths.cuda()
                example_labels = example_labels.cuda()
            # predicted shape: (batch_size, seq_len, 2)
            predicted = model(example_text, example_lengths)
            batch_loss = loss_criterion(predicted.view(-1, 2), example_labels.view(-1))
            rnn_optimizer.zero_grad()
            batch_loss.backward()
            rnn_optimizer.step()
            num_iter += 1
            # Get predictions, update confusion matrix
            _, predicted_labels = torch.max(predicted.data, 2)
            train_confusion_matrix = eval_util.update_confusion_matrix(train_confusion_matrix, predicted_labels,
                                                                       example_labels.data)
            # Calculate validation and training set loss and accuracy every 200 gradient updates
            if num_iter % print_every == 0:
                if verbose:
                    train_performance = eval_util.print_info(train_confusion_matrix)
                    train_confusion_matrix = np.zeros((2, 2))
                    print(f"Iteration {num_iter}")
                    print(f"Trn Performance: {train_performance}, Loss {batch_loss.item()}")

                if val_dataloader is not None:
                    avg_eval_loss, performance = eval_util.evaluate(val_dataloader, model, loss_criterion, using_GPU)
                    if performance[-1] > optimal_score[-1]:
                        optimal_score = performance
                        optimal_state_dict = model.state_dict()
                    if verbose:
                        print(f"Val Performance: {performance}, Loss {avg_eval_loss}")
                filename = f"models/{prefix}_iter_{num_iter}.pt"
                torch.save(model.state_dict(), filename)
    model.load_state_dict(optimal_state_dict)
    return model, optimal_score


# train_k_fold(5, raw_train)
# For one big training set. Runs k-fold validation.
def train_k_fold(k=5, train_data=raw_train, batch_size=16, num_epochs=20, print_every=200,
                 prefix=dataset):
    embedded_train, embedding_dim = embed_data(train_data, True)

    optimal_scores = []
    n = len(embedded_train)
    fold_size = int(n / k)

    # Change the seed to vary training fold assignment.
    inds = [i for i in range(n)]
    random.seed(0)
    random.shuffle(inds)

    for i in range(k):
        # Divide into train/validation
        val_indices = [inds[z] for z in range(i * fold_size, (i + 1) * fold_size)]
        fold_train = [embedded_train[j] for j in range(n) if j not in val_indices]
        fold_valid = [embedded_train[j] for j in range(n) if j in val_indices]
        print(f"Fold {i} train/test metaphorical word count:",
              sum([sum(ex[1]) for ex in fold_train]), sum([sum(ex[1]) for ex in fold_valid]))

        # Feed into Dataset -> DataLoader to train in batches.
        train_dataset = data_util.TextDataset([example[0] for example in fold_train],
                                              [example[1] for example in fold_train],
                                              [example[2] for example in fold_train])
        valid_dataset = data_util.TextDataset([example[0] for example in fold_valid],
                                              [example[1] for example in fold_valid],
                                              [example[2] for example in fold_valid])

        # For near-reproducible results do not use DataLoader shuffle.
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=data_util.TextDataset.collate_fn)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                      collate_fn=data_util.TextDataset.collate_fn)

        # Train the model for this fold
        clf, optimal_score = train_model(train_dataloader, valid_dataloader, embedding_dim,
                                         num_epochs, print_every, f"{prefix}_fold_{i}")
        optimal_scores.append(optimal_score)
    return optimal_scores


"""
Predictions and writing answers
"""


def predict(model, embedded_test):
    """ Return probabilities of positive class. """
    model.eval()
    id_to_pred_idx = {}
    preds = []
    idx = 0
    for (embedded_sentence, feature_sequence) in embedded_test:
        embedded_sentence = Variable(torch.Tensor([embedded_sentence]))
        embedded_len = Variable(torch.Tensor([embedded_sentence.shape[1]]))
        if using_GPU:
            embedded_sentence = embedded_sentence.cuda()
            embedded_len = embedded_len.cuda()
        predicted = model(embedded_sentence, embedded_len)
        predicted_proba = predicted.data[0]
        for i in range(len(feature_sequence)):
            id_to_pred_idx[feature_sequence[i]['id']] = idx
            preds.append(predicted_proba[i][1].item())
            idx += 1
    preds = np.exp(np.array(preds))
    model.train()
    return id_to_pred_idx, preds


def write_preds_toefl(id_to_pred_idx, preds, threshold):
    """
    id_to_pred_idx[id] = probability that word is metaphoric
    :param id_to_pred_idx: maps a token tuple ('essayid', sentence_idx, word_idx) to prediction index
    :param preds: nparray where ith row is positive label probability for the word
    :param threshold: preds value > threshold becomes 1, else 0
    :return:
    """
    ptoks = load_util_toefl.load_test_tokens()
    answers = []
    preds = np.where(preds > threshold, 1, 0)
    for ptok in ptoks:
        txt_id, sent_id, word_id = ptok.split("_")
        prediction = preds[id_to_pred_idx[(txt_id, int(sent_id), int(word_id))]]
        answers.append(f"{ptok},{prediction}\n")

    with open("answer.txt", "w") as ans:
        ans.writelines(answers)


def load_model(filename, embedding_dim):
    model = RNNSequenceModel(num_classes=2, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=1,
                             bidir=True, dropout1=dropouts[0], dropout2=dropouts[1], dropout3=dropouts[2])
    model.load_state_dict(torch.load(filename))
    if using_GPU:
        model.cuda()
    return model


def pred_and_write(test_data, filename, i, k):
    embedded_test, embedding_dim = embed_data(test_data, False)
    clf = load_model(filename, embedding_dim)
    threshold = get_model_threshold(clf, i, k)
    id_to_pred_idx, preds = predict(clf, embedded_test)
    write_preds_toefl(id_to_pred_idx, preds, threshold)


def print_report(scores):
    print()
    for row in scores:
        print("\t".join([str(col) for col in row]))


# Need to specify which fold out of k folds this model was trained on
def get_model_threshold(clf, i, k):
    fold_valid = get_ith_fold_embedded(raw_train, i, k)[1]
    val_y = np.concatenate([example[1] for example in fold_valid])  # Labels
    id_to_pred_idx, prob_y = predict(clf, [[example[0], example[2]] for example in fold_valid])
    return eval_util.get_optimal_threshold_f1(val_y, prob_y)


"""
Word-level models
"""

import sklearn.ensemble as ensemble
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.base import clone


# clf = LogisticRegression()
# clf = ensemble.GradientBoostingClassifier()
# clf = ensemble.RandomForestClassifier()


def train_raw_split_w(train_data=raw_train, valid_data=raw_valid):
    embed_train, embed_dim = embed_data(train_data, True)
    embed_valid, embed_dim = embed_data(valid_data, True)
    train_word_x, train_word_y, train_word_f, train_ids = data_util.seq_to_word_level_train(embed_train)
    valid_word_x, valid_word_y, valid_word_f, valid_ids = data_util.seq_to_word_level_train(embed_valid)
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(train_word_x, train_word_y)

    fold_prob_y = clf.predict_proba(valid_word_x)[:, 1]
    valid_threshold = eval_util.get_optimal_threshold_f1(valid_word_y, fold_prob_y)
    fold_pred_y = np.where(fold_prob_y > valid_threshold, 1, 0)

    # fold_train_prob = clf.predict_proba(train_word_x)[:, 1]
    # fold_train_pred = np.where(fold_train_prob > valid_threshold, 1, 0)

    # t_score = metrics.precision_recall_fscore_support(fold_train_y, fold_train_pred, labels=[1])
    # print("Train:", '\t'.join([str(score.item()) for score in t_score]))
    v_score = metrics.precision_recall_fscore_support(valid_word_y, fold_pred_y, labels=[1])
    print("Valid:\t" + '\t'.join([str(score.item()) for score in v_score]))
    return valid_word_f, valid_word_y, fold_pred_y


# clfs, ths = train_k_fold_w(clf=LogisticRegression(solver='lbfgs', max_iter=300, class_weight=class_weight), f=[])
# clfs, ths = train_k_fold_w()
@ignore_warnings(category=ConvergenceWarning)
def train_k_fold_w(clf=LogisticRegression(solver='lbfgs'), train_data=raw_train, k=5, seed=0):
    embedded_train, embedding_dim = embed_data(train_data, True)
    train_word_x, train_word_y, train_word_f, train_ids = data_util.seq_to_word_level_train(embedded_train)

    n = len(train_word_x)
    fold_size = int(n / k)

    indices = [i for i in range(n)]
    random.seed(seed)
    random.shuffle(indices)
    clfs, ths = [], []
    for i in range(k):
        val_indices = [indices[j] for j in range(i * fold_size, (i + 1) * fold_size)]
        fold_train_x = [train_word_x[j] for j in range(n) if j not in val_indices]
        fold_train_y = [train_word_y[j] for j in range(n) if j not in val_indices]
        fold_valid_x = [train_word_x[j] for j in range(n) if j in val_indices]
        fold_valid_y = [train_word_y[j] for j in range(n) if j in val_indices]

        fold_clf = clone(clf)
        fold_clf.fit(fold_train_x, fold_train_y)

        fold_prob_y = fold_clf.predict_proba(fold_valid_x)[:, 1]
        valid_threshold = eval_util.get_optimal_threshold_f1(fold_valid_y, fold_prob_y)
        fold_pred_y = np.where(fold_prob_y > valid_threshold, 1, 0)

        fold_train_prob = fold_clf.predict_proba(fold_train_x)[:, 1]
        fold_train_pred = np.where(fold_train_prob > valid_threshold, 1, 0)

        # t_score = metrics.precision_recall_fscore_support(fold_train_y, fold_train_pred, labels=[1])
        # print("Train:", '\t'.join([str(score.item()) for score in t_score]))
        v_score = metrics.precision_recall_fscore_support(fold_valid_y, fold_pred_y, labels=[1])
        print("Valid:\t" + '\t'.join([str(score.item()) for score in v_score]))

        clfs.append(fold_clf)
        ths.append(valid_threshold)
    return clfs, ths


def make_preds_th(clf, threshold, test_data):
    embedded_test, embedding_dim = embed_data(test_data, False)
    test_x, test_f, test_ids = data_util.seq_to_word_level_test(embedded_test)
    test_id_to_idx = {test_ids[i]: i for i in range(len(test_ids))}
    prob_y = clf.predict_proba(test_x)[:, 1]
    pred_y = np.where(prob_y < threshold, 0, 1)
    write_preds_toefl(test_id_to_idx, pred_y, threshold)


"""
Diagnostics
"""


def get_id_to_sentence_idx(raw_data):
    id_to_sentence_idx = {}
    for i in range(len(raw_data)):
        features_seq = raw_data[i][-1]
        id_to_sentence_idx[features_seq[0]['id'][:2]] = i
    return id_to_sentence_idx


# fold_train, fold_valid = data_util.get_ith_fold_raw(raw_train, 0, 5)
# disagree_examples(pred_y_lr, pred_y_rnn, fold_valid, fold_valid_f)
# print_report_by_feature('stanford_postag', fold_valid_f, fold_valid_y, pred_y_bl)
def disagree_examples(y1, y2, fold_valid, fold_valid_f):
    id_to_sentence_idx = get_id_to_sentence_idx(fold_valid)
    dis_idx = np.concatenate(np.argwhere(y1 != y2))
    for disagree_idx in dis_idx:
        features = fold_valid_f[disagree_idx]
        ex_id = features['id']
        label = features['label']
        s_pos = features['stanford_postag']
        s_idx = id_to_sentence_idx[ex_id[:2]]
        sentence = fold_valid[s_idx][0]
        word = sentence.split()[ex_id[2] - 1]
        print(word, s_pos, label, s_idx, ex_id[2] - 1)
        print(sentence)


def print_cm_2x2(cm):
    print(f"{cm[0][0]}\t{cm[0][1]}")
    print(f"{cm[1][0]}\t{cm[1][1]}")


# Only for POS tag
# fold_valid_f, fold_valid_y, pred_y = rnn_validation_preds(*get_ith_fold_embedded())
# print_report_by_feature('stanford_postag', fold_valid_f, fold_valid_y, pred_y)


def print_report_by_feature(feature_name, fold_valid_f, fold_valid_y, pred_y):
    inds_by_feature = {}
    for i in range(len(fold_valid_f)):
        feature_value = fold_valid_f[i][feature_name]
        if feature_value not in inds_by_feature:
            inds_by_feature[feature_value] = []
        inds_by_feature[feature_value].append(i)
    for feature_value in inds_by_feature:
        inds = inds_by_feature[feature_value]
        # print(metrics.confusion_matrix(fold_valid_y[inds], pred_y[inds]))
        acc = metrics.accuracy_score(fold_valid_y[inds], pred_y[inds])
        prf = metrics.precision_recall_fscore_support(fold_valid_y[inds], pred_y[inds], labels=[1])
        prf = [x.item() for x in prf]
        scores = [feature_value, str(acc)] + [str(x) for x in prf]
        print('\t'.join(scores))


def print_data_count(raw_data, feature_name):
    m_count_dict = {x: [0, 0] for x in data_util.get_feature_value_set(raw_data, feature_name)}
    for example in raw_data:
        features_seq = example[-1]
        for i in range(len(features_seq)):
            val = features_seq[i][feature_name]
            label = features_seq[i]['label']
            m_count_dict[val][label] += 1
    for key, value in sorted(m_count_dict.items(), key=lambda x: x[1][1]):
        print(key.rjust(20) + '\t' + str(value[0]) + '\t' + str(value[1]))


def find_matching_examples(feature_name, feature_value, raw_data):
    positive = set()
    negative = set()
    for i in range(len(raw_data)):
        example = raw_data[i]
        features_seq = example[-1]
        for j in range(len(features_seq)):
            features = features_seq[j]
            if features[feature_name] == feature_value:
                if features['label'] == 1:
                    positive.add((i, j))
                else:
                    negative.add((i, j))
    return positive, negative


def print_exs(examples, raw_data):
    for i, j in examples:
        example = raw_data[i]
        print(example[-1][j]['word'], ":", example[0])
        print([features['label'] for features in example[-1]])
