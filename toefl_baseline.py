import data_util
import eval_util
import load_util_toefl
import sklearn.metrics as metrics
import numpy as np
import random

# load word-level
raw_train, raw_test = load_util_toefl.load_train_test()
# raw_train, raw_test = load_util_toefl.load_from_pkl("TOEFL_clean")


def fit(fold_train_f, fold_train_y, pred_feature):
    m_count = {}
    for i in range(len(fold_train_f)):
        features = fold_train_f[i]
        label = fold_train_y[i]

        feature_value = features[pred_feature]
        if feature_value not in m_count:
            m_count[feature_value] = [0, 0]
        m_count[feature_value][label] += 1

        ul_value = features['ul']
        if ul_value not in m_count:
            m_count[ul_value] = [0, 0]
        m_count[ul_value][label] += 1
    return m_count


# UL fallback -> literal fallback
def predict_proba(m_count, fold_train_f, pred_feature):
    counts = [m_count.get(ex[pred_feature], m_count.get(ex['ul'], [1, 0])) for ex in fold_train_f]
    return np.array([count[1]/sum(count) for count in counts])


# mcs, ths = train_k_fold(raw_train, 5, 'ul')
def train_k_fold(train_data=raw_train, k=5, pred_feature='ul'):
    train_x, train_y, train_f, train_ids = data_util.seq_to_word_level_train(train_data)

    n = len(train_ids)
    fold_size = int(n / k)
    indices = [i for i in range(n)]
    random.seed(0)
    random.shuffle(indices)
    m_counts, thresholds = [], []
    for i in range(k):
        val_indices = [indices[j] for j in range(i * fold_size, (i + 1) * fold_size)]
        fold_train_f = [train_f[j] for j in range(n) if j not in val_indices]
        fold_train_y = [train_y[j] for j in range(n) if j not in val_indices]
        fold_valid_f = [train_f[j] for j in range(n) if j in val_indices]
        fold_valid_y = [train_y[j] for j in range(n) if j in val_indices]

        m_count = fit(fold_train_f, fold_train_y, pred_feature)

        fold_valid_prob = predict_proba(m_count, fold_valid_f, pred_feature)
        valid_threshold = eval_util.get_optimal_threshold_f1(fold_valid_y, fold_valid_prob)
        fold_valid_pred = np.where(fold_valid_prob > valid_threshold, 1, 0)

        fold_train_prob = predict_proba(m_count, fold_train_f, pred_feature)
        fold_train_pred = np.where(fold_train_prob > valid_threshold, 1, 0)

        t_scores = metrics.precision_recall_fscore_support(fold_train_y, fold_train_pred, labels=[1])
        print("Train:", '\t'.join([str(score.item()) for score in t_scores]))
        v_scores = metrics.precision_recall_fscore_support(fold_valid_y, fold_valid_pred, labels=[1])
        print("Valid:", '\t'.join([str(score.item()) for score in v_scores]))

        m_counts.append(m_count)
        thresholds.append(valid_threshold)
    return m_counts, thresholds


def pred_and_write(test_data, m_count, threshold, pred_feature):
    test_x, test_f, test_ids = data_util.seq_to_word_level_test(test_data)
    test_prob_y = predict_proba(m_count, test_f, pred_feature)
    test_pred_y = np.where(test_prob_y > threshold, 1, 0)

    id_to_pred = {test_ids[i]: test_pred_y[i] for i in range(len(test_ids))}
    ptoks = load_util_toefl.load_test_tokens()
    answers = []
    for ptok in ptoks:
        txt_id, sent_id, word_id = ptok.split("_")
        prediction = id_to_pred[(txt_id, int(sent_id), int(word_id))]
        answers.append(f"{ptok},{prediction}\n")

    with open("answer.txt", "w") as ans:
        ans.writelines(answers)


def print_report(metric_list):
    print(f"Results across {len(metric_list)}-fold validation:")
    print("Accuracy\tPrecision\tRecall\tF1")
    for row in metric_list:
        print("\t".join([str(col) for col in row]))
