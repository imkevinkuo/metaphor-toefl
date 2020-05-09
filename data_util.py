import os

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import mmap
import random
import pattern.text.en as pattern
import pickle as pkl


def get_vocab(raw_dataset):
    vocab = []
    for example in raw_dataset:
        vocab.extend(example[0].split())
    vocab = set(vocab)
    print("vocab size: ", len(vocab))
    return vocab


def get_feature_value_set(raw_dataset, feature_name):
    value_set = set()
    for sentence_seq, feature_seq in raw_dataset:
        value_set.update([features[feature_name] for features in feature_seq])
    return value_set


def get_pos2idx_idx2pos(vocab):
    """
    :param vocab: a set of strings: all pos tags
    :return: word2idx: a dictionary: string to an int
             idx2word: a dictionary: int to a string
    """
    word2idx = {}
    idx2word = {}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_glove_embedding_matrix(word2idx, idx2word, normalization=False):
    """
    assume padding index is 0

    :param word2idx: a dictionary: string --> int, includes <PAD> and <UNK>
    :param idx2word: a dictionary: int --> string, includes <PAD> and <UNK>
    :param normalization:
    :return: an embedding matrix: a nn.Embeddings
    """
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    embedding_dim = 300
    glove_path = "glove/glove840B300d.txt"
    glove_vectors = {}
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector

    print("Number of pre-trained word vectors loaded: ", len(glove_vectors))

    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)

    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings


def get_word2idx_idx2word(vocab):
    """

    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def embed_indexed_sequence(sentence, feature_seq, glove_embeddings, word2idx, elmo_embeddings, sequence_idx, other_features):
    """
    Assume that pos_seq maps well with sentence
    Assume that the given sentence is indexed by word2idx
    Assume that word2idx has 1 mapped to UNK
    Assume that word2idx maps well implicitly with glove_embeddings
    Assume that the given pos_seq is indexed by pos2idx
    Assume that pos2idx maps well implicitly with pos_embeddings
    i.e. the idx for each word is the row number for its corresponding embedding

    :param sentence        : raw sentence string
    :param feature_seq     : list of dicts: {feature: feature_value}
    :param glove_embeddings: nn.Embedding with padding idx 0, uses word2idx
    :param word2idx        : dictionary: string --> int
    :param elmo_embeddings : loaded .h5py file: maps sequence_idx string (e.g. '0') to np array (seq_len, 1024 elmo)
    :param sequence_idx    : the sequence's associated index for elmo vector
    :param other_features  : list of strings specifying other features to include from feature_seq
    :return: result          np.array (seq_len, embed_dim)
    """
    words = sentence.split()

    result = np.zeros((len(words), 0))
    glove_part = np.zeros((len(words), 0))
    elmo_part = np.zeros((len(words), 0))
    other_part = np.zeros((len(words), 0))

    # 1. glove embeddings
    if glove_embeddings is not None:
        # Replace words with tokens, and 1 (UNK index) if words not indexed.
        indexed_sequence = [word2idx.get(x, 1) for x in words]
        glove_part = glove_embeddings(torch.LongTensor(indexed_sequence)).data
        assert (glove_part.shape == (len(words), 300))

    # 2. elmo embeddings
    if elmo_embeddings is not None:
        elmo_part = elmo_embeddings[sequence_idx]
        assert (elmo_part.shape == (len(words), 1024))

    # 3. pos tags and other annotations
    for f_name in other_features:
        other_part = np.concatenate((other_part, np.stack([f_dict[f_name] for f_dict in feature_seq], axis=0)), axis=1)

    result = np.concatenate((result, glove_part, elmo_part, other_part), axis=1)
    assert (len(words) == result.shape[0])
    assert (len(feature_seq) == result.shape[0])
    return result


# Make sure to subclass torch.utils.data.Dataset
class TextDataset(Dataset):
    def __init__(self, embedded_text, label_seqs, feature_seqs):
        """

        :param embedded_text: A list of numpy arrays, each inner numpy array is sequence_length * embed_dim
        :param feature_seqs:  A list of lists, each inner list contains sequence_length dicts of word-level features.
        """
        if len(embedded_text) != len(feature_seqs):
            raise ValueError("Differing number of sentences and features!")

        self.embedded_text = embedded_text
        self.label_seqs = label_seqs
        self.feature_seqs = feature_seqs

    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.
        """
        example_text = self.embedded_text[idx]
        example_length = example_text.shape[0]
        example_label_seq = self.label_seqs[idx]
        example_feature_seq = self.feature_seqs[idx]
        assert (example_length == len(example_label_seq))
        return example_text, example_length, example_label_seq, example_feature_seq

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.embedded_text)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.
        """
        batch_padded_example_text = []
        batch_lengths = []
        batch_padded_labels = []
        batch_feature_seqs = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for text, __, __, __ in batch:
            if len(text) > max_length:
                max_length = len(text)

        # Iterate over each example in the batch
        for text, length, label, features in batch:
            # Unpack the example (returned from __getitem__)
            # append the pos_sequence to the batch_pos_seqs
            batch_feature_seqs.append(features)

            # Amount to pad is length of longest example - length of this example.
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])

            # Append the pad_tensor to the example_text tensor.
            # Shape of padded_example_text: (padded_length, embeding_dim)
            # top part is the original text numpy,
            # and the bottom part is the 0 padded tensors

            # text from the batch is a np array, but cat requires the argument to be the same type
            # turn the text into a torch.FloatTenser, which is the same type as pad_tensor
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # pad the labels with zero.
            padded_example_label = label + [0] * amount_to_pad

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_padded_labels.append(padded_example_label)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_padded_labels),
                batch_feature_seqs)


# For creating ELMo vectors.
# Run this function in the Python console, using the variables returned by load_train_test()
# Open elmo folder in terminal and use allennlp to write the ELMo vectors to disk.
# allennlp elmo TOEFL_clean_train.txt TOEFL_clean_train.hdf5 --average --cuda-device 0
def write_sentences_to_txt(raw_data, filename):
    sentences = set()
    for sentence, features_seq in raw_data:
        if sentence != '':
            sentences.add(sentence + '\n')
    with open(f"elmo/{filename}", 'w') as f:
        f.writelines(sentences)


def get_ith_fold_raw(train_data, i=0, k=5, seed=0):
    n = len(train_data)
    fold_size = int(n / k)

    inds = [j for j in range(n)]
    random.seed(seed)
    random.shuffle(inds)

    val_indices = [inds[j] for j in range(i * fold_size, (i + 1) * fold_size)]
    fold_train = [train_data[j] for j in range(n) if j not in val_indices]
    fold_valid = [train_data[j] for j in range(n) if j in val_indices]

    return fold_train, fold_valid


# todo
# Do not split sentences from same essay between train/val.
# attempts to sample essays equally across prompts.
# bin the essays by prompt, split each bin into k folds, generate fold lists containing essay ids


"""
To train with an augmented set of sentences (with errors injected):
1. Run data_util.py in the Python console, then run one of the train_inject_errors() functions.
2. use allennlp cli to write the contents of "elmo/..._AUG_train.txt" to an .hdf5 file.
3. Run toefl_lstm.py, using "TOEFL_AUG" or "VT_AUG" as the dataset. The test elmo file is the same as TOEFL_test.hdf5.
    Run train_k_fold or batch_and_train using raw_train and aug_train as parameters.
"""


def train_inject_errors_toefl(i=0, k=5, seed=0):
    """
    TODO: Separate based on English proficiency.
    TODO: Do not use chr errors, and spell check everything.
    1. separate a dev set from the training data (without a non-augmented dev set we are prone to overfitting)
    2. create erroneous copies of existing sentences
    3. pickle everything
    4. save sentences to txt for elmo vector creation
    """
    import load_util_toefl
    raw_train_all, raw_test = load_util_toefl.load_train_test()

    raw_train, raw_valid = get_ith_fold_raw(raw_train_all, i, k, seed)
    n = len(raw_train)
    # TODO: Get all DET pos, Get all ADP pos

    for i in tqdm(range(n)):
        err_copies = [inject_art_error(raw_train[i]),
                      inject_adp_error(raw_train[i]),
                      inject_nn_error(raw_train[i])]
        for err_copy in err_copies:
            if err_copy:
                raw_train.append(err_copy)

    pkl.dump(raw_train, open(f"{load_util_toefl.TOEFL_TRAIN}/TOEFL_AUG_train.pkl", "wb"))
    pkl.dump(raw_valid, open(f"{load_util_toefl.TOEFL_TRAIN}/TOEFL_AUG_valid.pkl", "wb"))
    pkl.dump(raw_test, open(f"{load_util_toefl.TOEFL_TRAIN}/TOEFL_AUG_test.pkl", "wb"))
    write_sentences_to_txt(raw_train + raw_valid, "TOEFL_AUG_train.txt")


# Creates and saves an augmented set of VUA sentences
# Maybe also filter on genre?
# We dont need to separate a validation set because we'll be validating on TOEFL
def train_inject_errors_vt(seed):
    import load_util_vua
    import load_util_toefl
    random.seed(0)
    raw_train_vua, raw_test_vua = load_util_vua.load_train_test()
    #
    raw_train_vua = load_util_vua.filter_train_by_genre(raw_train_vua, ['academic'])
    raw_train_vua = load_util_vua.filter_by_length(raw_train_vua, 4)
    if len(raw_train_vua) > 2000:
        raw_train_vua = random.sample(raw_train_vua, 2000)
    #
    n = len(raw_train_vua)
    aug_data = []
    for i in tqdm(range(n)):
        x = random.random()
        if x < 0.25:
            aug_data.append(inject_chr_error(raw_train_vua[i]))
        elif 0.25 <= x < 0.5:
            aug_data.append(inject_art_error(raw_train_vua[i]))
        elif 0.5 <= x < 0.75:
            aug_data.append(inject_adp_error(raw_train_vua[i]))
        else:
            aug_data.append(inject_nn_error(raw_train_vua[i]))
        # raw_train.append(inject_sva_error(raw_train_vua[i]))

    pkl.dump(aug_data, open(f"{load_util_toefl.TOEFL_TRAIN}/VT_AUG_train.pkl", "wb"))
    raw_train_toefl, raw_test_toefl = load_util_toefl.load_train_test()
    write_sentences_to_txt(aug_data + raw_train_toefl, "VT_AUG_train.txt")


# Character error: delete random character
def inject_chr_error(example):
    tok_sentence, features_seq = example[0].split(), example[1]
    word_indices = []
    for i in range(len(tok_sentence)):
        if len(tok_sentence[i]) > 3:
            word_indices.append(i)
    if len(word_indices) > 0:
        j = random.choice(word_indices)  # choose random word with length > 3
        k = random.randint(1, len(tok_sentence[j]) - 2)  # choose random character
        tok_sentence[j] = tok_sentence[j][:k] + tok_sentence[j][k+1:]
    return [' '.join(tok_sentence), features_seq]


# Article error: In universal POS, DET includes articles and pro-adjs
def inject_art_error(example):
    # Random deletion
    tok_sentence, features_seq = example[0].split(), example[1]
    det_indices = []
    for i in range(len(features_seq)):
        if features_seq[i]['universal_postag'] == 'DET':
            det_indices.append(i)

    if len(det_indices) > 0:
        j = random.choice(det_indices)
        tok_sentence = tok_sentence[:j] + tok_sentence[j + 1:]
        features_seq = features_seq[:j] + features_seq[j + 1:]
    else:
        return None
    return [' '.join(tok_sentence), features_seq]


# Adposition: includes pre/postposition
def inject_adp_error(example):
    # Random deletion
    tok_sentence, features_seq = example[0].split(), example[1]
    det_indices = []
    for i in range(len(features_seq)):
        if features_seq[i]['universal_postag'] == 'ADP':
            det_indices.append(i)

    if len(det_indices) > 0:
        j = random.choice(det_indices)
        tok_sentence = tok_sentence[:j] + tok_sentence[j + 1:]
        features_seq = features_seq[:j] + features_seq[j + 1:]
    else:
        return None
    return [' '.join(tok_sentence), features_seq]


# Noun number
def inject_nn_error(example):
    tok_sentence, features_seq = example[0].split(), example[1]
    noun_indices = []
    for i in range(len(features_seq)):
        if features_seq[i]['universal_postag'] == 'NOUN':
            noun_indices.append(i)

    if len(noun_indices) > 0:
        j = random.choice(noun_indices)
        singular_form = pattern.singularize(tok_sentence[j])
        plural_form = pattern.pluralize(tok_sentence[j])
        if tok_sentence[j] != singular_form:
            tok_sentence[j] = singular_form
        elif tok_sentence[j] != plural_form:
            tok_sentence[j] = plural_form
    else:
        return None
    return [' '.join(tok_sentence), features_seq]


# Subject-verb agreement
# Doesn't work on Python 3.7
def inject_sva_error(example):
    tok_sentence, features_seq = example[0].split(), example[1]
    verb_indices = []
    for i in range(len(features_seq)):
        if features_seq[i]['universal_postag'] == 'VERB':
            verb_indices.append(i)

    if len(verb_indices) > 0:
        j = random.choice(verb_indices)
        lexeme = pattern.lexeme(tok_sentence[j])
        if len(lexeme) > 0:
            tok_sentence[j] = random.choice(lexeme)

    return [' '.join(tok_sentence), features_seq]


def seq_to_word_level_train(train_embedded):
    train_word_x = []
    train_word_y = []
    train_word_f = []
    train_ids = []

    for ex in train_embedded:
        embedded_seq = ex[0]
        features_seq = ex[-1]
        for i in range(len(features_seq)):
            if features_seq[i]['stanford_postag'] != 'UNK':
                train_word_x.append(embedded_seq[i])
                train_word_y.append(features_seq[i]['label'])
                train_word_f.append(features_seq[i])
                train_ids.append(features_seq[i]['id'])

    train_word_x, train_word_y = np.array(train_word_x), np.array(train_word_y)
    return train_word_x, train_word_y, train_word_f, train_ids


def seq_to_word_level_test(test_embedded):
    test_word_x = []
    test_word_f = []
    test_ids = []

    for embedded_seq, features_seq in test_embedded:
        for i in range(len(features_seq)):
            if features_seq[i]['stanford_postag'] != 'UNK':
                test_word_x.append(embedded_seq[i])
                test_word_f.append(features_seq[i])
                test_ids.append(features_seq[i]['id'])

    test_word_x = np.array(test_word_x)
    return test_word_x, test_word_f, test_ids


def dependency_parse(raw_data):
    from nltk.parse.corenlp import CoreNLPServer

    # The server needs to know the location of the following files:
    #   - stanford-corenlp-X.X.X.jar
    #   - stanford-corenlp-X.X.X-models.jar
    STANFORD = os.path.join("..", "stanford-corenlp-full-2020-04-20")

    # Create the server
    server = CoreNLPServer(
        os.path.join(STANFORD, "stanford-corenlp-4.0.0.jar"),
        os.path.join(STANFORD, "stanford-corenlp-4.0.0-models.jar"),
    )

    # Start the server in the background
    server.start()
    from nltk.parse import CoreNLPParser
    parser = CoreNLPParser()

    new_data = []
    for example in raw_data:
        sentence, features_seq = example[0], example[-1]
        parse = next(parser.raw_parse(sentence))
        # get a few "important" neighboring words

    server.stop()

