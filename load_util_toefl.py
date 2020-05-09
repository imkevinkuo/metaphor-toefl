import os
from copy import copy

import nltk
import jsonlines
import numpy as np
from tqdm import tqdm
import pickle as pkl
import data_util

TOEFL_TRAIN = "data/toefl_sharedtask_dataset"
TOEFL_TEST = "data/toefl_sharedtask_evaluation_kit"
TOEFL_TEST_TOKENS = f"{TOEFL_TEST}/toefl_all_pos_test_tokens.csv"
TOEFL_TRAIN_FEATURES = f"{TOEFL_TRAIN}/toefl_skll_train_features/all_pos"
TOEFL_TEST_FEATURES = f"{TOEFL_TEST}/toefl_skll_test_features_no_labels/all_pos"


def load_train_test_no_f():
    raw_train, raw_test = [], []

    for directory in (TOEFL_TRAIN, TOEFL_TEST):
        if directory == TOEFL_TRAIN:
            ex_features = load_annotations(TOEFL_TRAIN_FEATURES, True)
            raw_data = raw_train
        else:
            ex_features = load_annotations(TOEFL_TEST_FEATURES, False)
            raw_data = raw_test

        sentences = load_sentences(directory)

        # txt_id is essay id
        for txt_id, sent_id in tqdm(sentences):
            sentence_txt = sentences[(txt_id, sent_id)].replace('M_', '')
            if sentence_txt != '':
                sentence_seq = [word for word in sentence_txt.split()]
                features_seq = []
                for i in range(len(sentence_seq)):
                    word_id = i + 1
                    txt_sent_word_id = (txt_id, sent_id, word_id)
                    if txt_sent_word_id in ex_features:
                        features = ex_features[txt_sent_word_id]
                    else:
                        features = {
                            'id': txt_sent_word_id
                        }
                        if directory == TOEFL_TRAIN:
                            features['label'] = 0

                    features['txt_id'] = txt_id
                    features_seq.append(features)

                raw_data.append([sentence_txt, features_seq])

    return raw_train, raw_test


def load_train_test():
    """
    Returns raw_data: list of [[sentence, feature_sequence]]
    """

    raw_train, raw_test = [], []

    for directory in (TOEFL_TRAIN, TOEFL_TEST):
        if directory == TOEFL_TRAIN:
            ex_features = load_annotations(TOEFL_TRAIN_FEATURES, True)
            raw_data = raw_train
        else:
            ex_features = load_annotations(TOEFL_TEST_FEATURES, False)
            raw_data = raw_test

        lemmatizer = nltk.WordNetLemmatizer()
        sentences = load_sentences(directory)
        essay_to_prompt = get_essay_to_prompt(directory)
        essay_to_native_lang, essay_to_proficiency = get_essay_to_lang_and_prof(directory)

        # txt_id is essay id
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
                            'txt_id': txt_id,
                            'ul': lemmatizer.lemmatize(word),
                            'word': word,
                            'stanford_postag': 'UNK',
                            'wordnet': np.zeros(15),
                            'topic_lda': np.zeros(100),
                            'cbiasup': np.zeros(17),
                            'cbiasdown': np.zeros(17),
                            'cbiasdiff': np.zeros(66)
                        }
                        if directory == TOEFL_TRAIN:
                            features['label'] = 0

                    features['word'] = word
                    # Features that do not come from annotations
                    features['universal_postag'] = univ_postags[i]

                    # one_hot_prompt = np.zeros(8)

                    features['txt_id'] = txt_id
                    features['prompt'] = essay_to_prompt[txt_id]

                    # one_hot_prompt[features['prompt']-1] = 1
                    # features['prompt_vec'] = one_hot_prompt

                    native_lang_indexer = {"ARA": 0, "ITA": 1, "JPN": 2}
                    proficiency_indexer = {"medium": 0, "high": 1}

                    features['native_lang'] = essay_to_native_lang[txt_id]
                    features['proficiency'] = essay_to_proficiency[txt_id]
                    one_hot_native_lang = np.zeros(3)
                    one_hot_proficiency = np.zeros(2)
                    one_hot_native_lang[native_lang_indexer[features['native_lang']]] = 1
                    one_hot_proficiency[proficiency_indexer[features['proficiency']]] = 1
                    features['native_lang_vec'] = one_hot_native_lang
                    features['proficiency_vec'] = one_hot_proficiency

                    features['bl_x'] = features['ul'] + str(features['prompt']) + str(features['proficiency'])

                    # Append dict to list
                    features_seq.append(features)

                raw_data.append([sentence_txt, features_seq])

    # Feature transformations that require entire dataset
    # One-hot encoding on Unigram Lemma
    ul_set = data_util.get_feature_value_set(raw_train + raw_test, 'ul')
    ul_dict = {}
    i = 0
    for lemma in ul_set:
        ul_dict[lemma] = i
        i += 1
    ul_count = len(ul_dict)
    for sentence, feature_seq in raw_train + raw_test:
        for features in feature_seq:
            ul_vector = np.zeros(ul_count)
            ul_vector[ul_dict[features['ul']]] = 1
            features['ul_vec'] = ul_vector


    # POS Embeddings
    universal_pos_set = data_util.get_feature_value_set(raw_train + raw_test, 'universal_postag')
    stanford_pos_set = data_util.get_feature_value_set(raw_train + raw_test, 'stanford_postag')
    universal_pos_count = len(universal_pos_set)
    stanford_pos_count = len(stanford_pos_set)

    # Universal POS has 12 tags, tagged by nltk
    pos2idx_univ, idx2pos_univ = data_util.get_pos2idx_idx2pos(universal_pos_set)
    for sentence, feature_seq in raw_train + raw_test:
        for features in feature_seq:
            pos_vector = np.zeros(universal_pos_count)
            pos_vector[pos2idx_univ[features['universal_postag']]] = 1
            features['universal_posvec'] = pos_vector

    # Stanford POS has 16 tags + UNK (for non-annotated words), annotated in data
    pos2idx_stan, idx2pos_stan = data_util.get_pos2idx_idx2pos(stanford_pos_set)
    for sentence, feature_seq in raw_train + raw_test:
        for features in feature_seq:
            pos_vector = np.zeros(stanford_pos_count)
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
def load_base_annotations(ex_features, directory, include_labels=True):
    with jsonlines.open(f"{directory}/UL.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            ex_id = (txt_id, int(sent_id), int(word_id))
            (ul, _), = obj["x"].items()
            ex_features[ex_id] = {}
            ex_features[ex_id]['ul'] = ul.split("_")[1]
            ex_features[ex_id]['id'] = ex_id
            if include_labels:
                ex_features[ex_id]['label'] = int(obj['y'])
    return ex_features


def load_pos_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/P.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            stanford_postag = obj['x']['stanford_postag']
            ex_features[(txt_id, int(sent_id), int(word_id))]['stanford_postag'] = stanford_postag
    return ex_features


def load_wordnet_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/WordNet.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            wn_vector = np.zeros(15)
            for annotation in obj['x']:
                index = int(annotation.split("_")[-1])
                wn_vector[index-1] = 1
            ex_features[(txt_id, int(sent_id), int(word_id))]['wordnet'] = wn_vector
    return ex_features


def load_topic_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/T.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            lda_vector = np.zeros(100)
            for annotation in obj['x']:
                index = int(annotation.split("-")[-1])
                lda_vector[index-1] = float(obj['x'][annotation])
            ex_features[(txt_id, int(sent_id), int(word_id))]['topic_lda'] = lda_vector
    return ex_features


def load_cbias_annotations(ex_features, directory):
    with jsonlines.open(f"{directory}/C-BiasUp.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            ccb_vector = np.zeros(17)
            for annotation in obj['x']:
                index = int(annotation.split("_")[-1])
                ccb_vector[index-1] = int(obj['x'][annotation])
            ex_features[(txt_id, int(sent_id), int(word_id))]['cbiasup'] = ccb_vector

    with jsonlines.open(f"{directory}/C-BiasDown.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            ccb_vector = np.zeros(17)
            for annotation in obj['x']:
                index = int(annotation.split("_")[-1])
                ccb_vector[index-1] = int(obj['x'][annotation])
            ex_features[(txt_id, int(sent_id), int(word_id))]['cbiasdown'] = ccb_vector

    with jsonlines.open(f"{directory}/CCDB-BiasUpDown.jsonlines") as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            ccb_vector = np.zeros(66)
            for annotation in obj['x']:
                index = int(annotation.split("_")[-1])
                ccb_vector[index-1] = int(obj['x'][annotation])
            ex_features[(txt_id, int(sent_id), int(word_id))]['cbiasdiff'] = ccb_vector
    return ex_features


def load_annotations(directory, include_labels):
    ex_features = {}
    load_base_annotations(ex_features, directory, include_labels)
    load_pos_annotations(ex_features, directory)
    load_wordnet_annotations(ex_features, directory)
    load_topic_annotations(ex_features, directory)
    load_cbias_annotations(ex_features, directory)
    return ex_features


def load_sentences(directory):
    sentences = {}
    for filename in os.listdir(f'{directory}/essays'):
        fileid = filename.split('.')[0]
        with open(f'{directory}/essays/{filename}') as f:
            lines = [line.rstrip() for line in f]
            for i in range(len(lines)):
                sentences[(fileid, i + 1)] = lines[i]
    return sentences


def load_test_tokens():
    test_toks = []
    with open(TOEFL_TEST_TOKENS) as f:
        lines = [line.rstrip() for line in f]
        for line in lines:
            txt_id, sent_id, verb_id, verb = line.split("_")
            test_toks.append("_".join((txt_id, sent_id, verb_id)))
    return test_toks


# From baseline
def get_m_count(train_set, pred_feature):
    """ Gets metaphoric/non-metaphoric occurrence count of each word. """
    m_count = {}
    for ex in train_set:
        features_seq = ex[-1]
        for i in range(len(features_seq)):
            feature = features_seq[i][pred_feature]
            if feature not in m_count:
                m_count[feature] = [0, 0]
            m_count[feature][features_seq[i]['label']] += 1
    return m_count


def get_essay_to_prompt(directory):
    essay_to_prompt = {}
    with open(f"{directory}/promptid_essayid.lst") as f:
        lines = f.read().splitlines()
        for line in lines:
            prompt_id, essay_id = line.split(",")
            essay_to_prompt[essay_id] = int(prompt_id[1])
    return essay_to_prompt


def get_essay_to_lang_and_prof(directory):
    essay_to_native_lang, essay_to_proficiency = {}, {}
    if directory == TOEFL_TRAIN:
        csv_file = "train_metadata.csv"
    else:
        csv_file = "test_metadata.csv"
    with open(f"{directory}/{csv_file}") as f:
        lines = f.read().splitlines()[1:]
        for line in lines:
            essay_id, native_lang, proficiency = line.split(",")
            essay_to_native_lang[essay_id] = native_lang
            essay_to_proficiency[essay_id] = proficiency
    return essay_to_native_lang, essay_to_proficiency


def load_from_pkl(dataset):
    raw_train = pkl.load(open(f"{TOEFL_TRAIN}/{dataset}_train.pkl", 'rb'))
    raw_test = pkl.load(open(f"{TOEFL_TRAIN}/{dataset}_test.pkl", 'rb'))
    return raw_train, raw_test


def load_data_aug(dataset):
    raw_train = pkl.load(open(f"{TOEFL_TRAIN}/{dataset}_train.pkl", 'rb'))
    raw_valid = pkl.load(open(f"{TOEFL_TRAIN}/{dataset}_valid.pkl", 'rb'))
    raw_test = pkl.load(open(f"{TOEFL_TRAIN}/{dataset}_test.pkl", 'rb'))
    return raw_train, raw_valid, raw_test


def seq_to_word(train_data, test_data):
    """ For use on non-embedded data. Converts sequential train/test data into word-level data for sklearn.
    Only preserve annotated train words - can detect since we gave them a value of UNK for stanford_postag
    Since Elmo vectors require the entire sentence, the function to actually use is in data_util."""

    train_word_level = []
    for sentence, features_seq in train_data:
        tok_sentence = sentence.split()
        for i in range(len(features_seq)):
            if features_seq['stanford_postag'] != 'UNK':
                train_word_level.append([tok_sentence[i], features_seq[i]])

    test_word_level = []
    for sentence, features_seq in test_data:
        tok_sentence = sentence.split()
        for i in range(len(features_seq)):
            if features_seq['stanford_postag'] != 'UNK':
                test_word_level.append([tok_sentence[i], features_seq[i]])

    return train_word_level, test_word_level


def filter_raw_data(raw_data):
    # Remove the features we generated
    new_raw_data = []
    for ex in raw_data:
        tok_sentence = ex[0].split()
        features_seq = ex[-1]
        keep_indices = [i for i in range(len(features_seq)) if features_seq[i]["stanford_postag"] != "UNK"]
        if len(keep_indices) > 0:
            new_raw_data.append([' '.join([tok_sentence[i] for i in keep_indices]),
                                 [features_seq[i] for i in keep_indices]])
        else:
            print("Discard whole sentence with no annotations:")
            print(ex[0])
    return new_raw_data


# raw_train, raw_test = load_train_test()
# spellcheck_all(raw_train, raw_test)
def spellcheck_all(raw_train, raw_test):
    """
    spellcheck takes a long time. process everything, write to text for ELMo creation, pickle the data.
    TOEFL_clean

    also lowercase everything

    Sometimes, joined words get split apart (which should be good), but we then need to fill in features...
    """
    from spellchecker import SpellChecker

    spell = SpellChecker()
    lemmatizer = nltk.WordNetLemmatizer()

    train_misspellings, test_misspellings = 0, 0

    for i in tqdm(range(len(raw_train))):
        ex = raw_train[i]
        tok_sentence = ex[0].split()
        features_seq = ex[-1]
        new_tok_sentence = []
        new_features_seq = []
        for j in range(len(tok_sentence)):
            new_words = spell.correction(tok_sentence[j]).split()  # Data contains stuff like "hisher", "inaddition"
            new_features_split = []  # copy the same feature set onto each word, then replace some values
            for k in range(len(new_words)):
                if new_words[k] != tok_sentence[j]:
                    train_misspellings += 1
                new_words[k] = new_words[k].lower()
                features = copy(features_seq[j])
                features['word'] = new_words[k]
                features['ul'] = lemmatizer.lemmatize(new_words[k])
                new_features_split.append(features)
            new_tok_sentence += new_words
            new_features_seq += new_features_split
        ex[0] = ' '.join(new_tok_sentence)
        ex[-1] = new_features_seq

    for i in tqdm(range(len(raw_test))):
        ex = raw_test[i]
        tok_sentence = ex[0].split()
        features_seq = ex[-1]
        new_tok_sentence = []
        new_features_seq = []
        for j in range(len(tok_sentence)):
            new_words = spell.correction(tok_sentence[j]).split()  # Data contains stuff like "hisher", "inaddition"
            new_features_split = []  # copy the same feature set onto each word, then replace some values
            for k in range(len(new_words)):
                if new_words[k] != tok_sentence[j]:
                    test_misspellings += 1
                new_words[k] = new_words[k].lower()
                features = copy(features_seq[j])
                features['word'] = new_words[k]
                features['ul'] = lemmatizer.lemmatize(new_words[k])
                new_features_split.append(features)
            new_tok_sentence += new_words
            new_features_seq += new_features_split
        ex[0] = ' '.join(new_tok_sentence)
        ex[-1] = new_features_seq

    print(train_misspellings, test_misspellings)
    pkl.dump(raw_train, open(f"{TOEFL_TRAIN}/TOEFL_clean_train.pkl", "wb"))
    pkl.dump(raw_test, open(f"{TOEFL_TRAIN}/TOEFL_clean_test.pkl", "wb"))

    data_util.write_sentences_to_txt(raw_train, "TOEFL_clean_train.txt")
    data_util.write_sentences_to_txt(raw_test, "TOEFL_clean_test.txt")
