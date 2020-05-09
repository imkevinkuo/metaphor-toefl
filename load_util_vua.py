import csv
import os
import nltk
import numpy as np
from jsonlines import jsonlines
from tqdm import tqdm

import data_util

VUA_TRAIN = "data/VUA_corpus/vuamc_corpus_train.csv"
VUA_TEST = "data/VUA_corpus/vuamc_corpus_test.csv"
VUA_TEST_TOKENS = f"data/VUA_corpus/all_pos_tokens_test.csv"
VUA_TRAIN_FEATURES = "data/VUA_corpus/naacl_flp_skll_train_datasets/all_pos"
VUA_TEST_FEATURES = "data/VUA_corpus/naacl_flp_skll_test_datasets/all_pos"


def load_train_test():
    """
    Returns raw_data: list of [[sentence, feature_sequence]]
    """

    raw_train, raw_test = [], []

    for directory in (VUA_TRAIN, VUA_TEST):
        if directory == VUA_TRAIN:
            ex_features = load_annotations(VUA_TRAIN_FEATURES, True)
            raw_data = raw_train
        else:
            ex_features = load_annotations(VUA_TEST_FEATURES, False)
            raw_data = raw_test

        lemmatizer = nltk.WordNetLemmatizer()
        sentences = load_sentences(directory)

        for txt_id, sent_id in tqdm(sentences):
            sentence_txt = sentences[(txt_id, sent_id)].replace('M_', '')
            if sentence_txt != '':
                sentence_seq = [word for word in sentence_txt.split()]
                features_seq = []
                univ_postags = [nltk.tag.map_tag('en-ptb', 'universal', tag) for _, tag in nltk.pos_tag(sentence_seq)]
                for i in range(len(sentence_seq)):
                    word = sentence_seq[i]
                    word_id = i + 1
                    txt_sent_word_id = (txt_id, sent_id, word_id)
                    # Reference the baseline features
                    if txt_sent_word_id in ex_features:
                        features = ex_features[txt_sent_word_id]
                    # If the word isn't annotated, then use the following defaults.
                    else:
                        features = {
                            'id': txt_sent_word_id,
                            'ul': lemmatizer.lemmatize(word),
                            'stanford_postag': 'UNK',
                            'wordnet': np.zeros(26),
                            'topic_lda': np.zeros(100),
                            'cbiasup': np.zeros(17),
                            'cbiasdown': np.zeros(17)
                        }
                        if directory == VUA_TRAIN:
                            features['label'] = 0

                    # Features that do not come from annotations
                    features['universal_postag'] = univ_postags[i]

                    # Append dict to list
                    features_seq.append(features)

                raw_data.append([sentence_txt, features_seq])

    # Feature transformations that require entire dataset

    # POS Embeddings
    universal_pos_set = data_util.get_feature_value_set(raw_train + raw_test, 'universal_postag')
    stanford_pos_set = data_util.get_feature_value_set(raw_train + raw_test, 'stanford_postag')
    univ_pos_count = len(universal_pos_set)
    stan_pos_count = len(stanford_pos_set)
    # Universal POS has 12 tags, tagged by nltk
    pos2idx_univ, idx2pos_univ = data_util.get_pos2idx_idx2pos(universal_pos_set)
    for sentence, feature_seq in raw_train + raw_test:
        for features in feature_seq:
            pos_vector = np.zeros(univ_pos_count)
            pos_vector[pos2idx_univ[features['universal_postag']]] = 1
            features['universal_posvec'] = pos_vector

    # Stanford POS has 16 tags + UNK (for non-annotated words), annotated in data
    pos2idx_stan, idx2pos_stan = data_util.get_pos2idx_idx2pos(stanford_pos_set)
    for sentence, feature_seq in raw_train + raw_test:
        for features in feature_seq:
            pos_vector = np.zeros(stan_pos_count)
            pos_vector[pos2idx_stan[features['stanford_postag']]] = 1
            features['stanford_posvec'] = pos_vector

    # Meta. vs literal occurrences
    m_count = get_m_count(raw_train, 'ul')
    for sentence, feature_seq in raw_train + raw_test:
        for features in feature_seq:
            m = m_count.get(features['ul'], [0, 0])
            features['m_count'] = np.array(m)

    return raw_train, raw_test


# Always load this one.
def load_label_ul_annotations(ex_features, directory, include_labels=True):
    with jsonlines.open(f"{directory}/UL.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id = obj['id'].split("_")
            ex_id = (txt_id, sent_id, int(word_id))
            (_, ul), = obj["x"].items()
            ex_features[ex_id] = {}
            ex_features[ex_id]['ul'] = ul
            ex_features[ex_id]['id'] = ex_id
            if include_labels:
                label = int(obj['y'])
                ex_features[ex_id]['label'] = label
    return ex_features


def load_pos_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/P.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id = obj['id'].split("_")
            stanford_postag = obj['x']['postag']
            ex_features[(txt_id, sent_id, int(word_id))]['stanford_postag'] = stanford_postag
    return ex_features


def load_wordnet_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/WordNet.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id = obj['id'].split("_")
            wn_vector = np.zeros(26)
            for annotation in obj['x']:
                index = int(annotation.split("_")[-1])
                wn_vector[index-1] = 1
            ex_features[(txt_id, sent_id, int(word_id))]['wordnet'] = wn_vector
    return ex_features


def load_topic_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/T.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id = obj['id'].split("_")
            lda_vector = np.zeros(100)
            for annotation in obj['x']:
                index = int(annotation.split("-")[-1])
                lda_vector[index-1] = float(obj['x'][annotation])
            ex_features[(txt_id, sent_id, int(word_id))]['topic_lda'] = lda_vector
    return ex_features


def load_cbias_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/C-BiasUp.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id = obj['id'].split("_")
            ccb_vector = np.zeros(17)
            for annotation in obj['x']:
                index = int(annotation.split("-")[-1])
                ccb_vector[index-1] = int(obj['x'][annotation])
            ex_features[(txt_id, sent_id, int(word_id))]['cbiasup'] = ccb_vector

    with jsonlines.open(f"{directory}/C-BiasDown.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id = obj['id'].split("_")
            ccb_vector = np.zeros(17)
            for annotation in obj['x']:
                index = int(annotation.split("-")[-1])
                ccb_vector[index-1] = int(obj['x'][annotation])
            ex_features[(txt_id, sent_id, int(word_id))]['cbiasdown'] = ccb_vector
    return ex_features


def load_annotations(directory, include_labels, genres=None):
    ex_features = {}
    dataset = "train" if directory == VUA_TRAIN_FEATURES else "test"
    # If none, do all of them
    if not genres:
        genres = os.listdir(directory)
    for genre in genres:
        genre_dir = os.path.join(directory, genre, dataset)
        load_label_ul_annotations(ex_features, genre_dir, include_labels)
        load_pos_annotations(ex_features, genre_dir)
        load_wordnet_annotations(ex_features, genre_dir)
        load_topic_annotations(ex_features, genre_dir)
        load_cbias_annotations(ex_features, genre_dir)
    return ex_features


def load_sentences(directory):
    sentences = {}
    with open(directory) as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            if len(line) > 0:
                sentences[(line[0], line[1])] = line[2]
    return sentences


def load_test_tokens():
    test_toks = []
    with open(VUA_TEST_TOKENS) as f:
        lines = [line.rstrip() for line in f]
        for line in lines:
            txt_id, sent_id, verb_id, verb = line.split("_")
            test_toks.append("_".join((txt_id, sent_id, verb_id)))
    return test_toks


# From baseline
def get_m_count(train_set, pred_feature):
    """ Gets metaphoric/non-metaphoric occurrence count of each word. """
    m_count = {}
    for sentence_txt, features_seq in train_set:
        for i in range(len(features_seq)):
            feature = features_seq[i][pred_feature]
            if feature not in m_count:
                m_count[feature] = [0, 0]
            m_count[feature][features_seq[i]['label']] += 1
    return m_count


def filter_train_by_genre(raw_data, genres):
    new_raw_data = []
    ex_features = load_annotations(VUA_TRAIN_FEATURES, False, genres)
    for sentence, features_seq in raw_data:
        if any([features['id'] in ex_features for features in features_seq]):
            new_raw_data.append((sentence, features_seq))
    return new_raw_data


def filter_test_by_genre(raw_data, genres):
    new_raw_data = []
    ex_features = load_annotations(VUA_TEST_FEATURES, False, genres)
    for sentence, features_seq in raw_data:
        if any([features['id'] in ex_features for features in features_seq]):
            new_raw_data.append((sentence, features_seq))
    return new_raw_data


def filter_by_length(raw_data, min_length):
    new_raw_data = []
    for sentence, features_seq in raw_data:
        if len(features_seq) > min_length:
            new_raw_data.append((sentence, features_seq))
    return new_raw_data
