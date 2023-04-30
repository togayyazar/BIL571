import numpy as np
from scipy.linalg import orthogonal_procrustes
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import requests
import json
import pickle as pkl
import gc
import time
from scipy.interpolate import CubicSpline
import pandas as pd

WORDNET_TAG_MAP = {
    'n': '_NOUN',
    's': '_ADJ',
    'a': '_ADJ',
    'r': '_ADV',
    'v': '_VERB'
}

lemmatize = False
LEMMATIZER = WordNetLemmatizer()

with open('data/sgns/1800-vocab.pkl', 'rb') as f:
    ORIG_VOCAB = pkl.load(f)
    vocab = []
    if lemmatize:
        for w in ORIG_VOCAB:
            vocab.append(LEMMATIZER.lemmatize(w))
    else:
        vocab = ORIG_VOCAB

VOCAB = vocab


def k_nearest_vectors(X, x, k):
    distances = []
    for i in range(X.shape[0]):
        distances.append(np.linalg.norm(X[i] - x))

    distances = np.array(distances)
    min_indices = np.argsort(distances)[0:k]

    return min_indices, X[min_indices, :]


def centroid(X):
    return np.mean(X, axis=0)


def monosemous_synsets(pos_tag=None):
    m_syns = []
    for synset in list(wn.all_synsets(pos_tag)):
        if len(synset.lemmas()) <= 1:
            continue

        ok = True
        for lemma in synset.lemmas():
            if len(wn.synsets(lemma.name(), synset.pos())) > 1:
                ok = False
                break
        if ok:
            m_syns.append(synset)

    return m_syns


def monosemous_words(pos_tag='n'):
    synsets = monosemous_synsets(pos_tag)
    words = {}
    for syn in synsets:
        all_unigram = True
        for lemma in syn.lemmas():
            if '_' in lemma.name():
                all_unigram = False
                break

        if not all_unigram:
            continue

        for lemma in syn.lemmas():
            words[lemma.name().lower()] = syn

    return words


def competitive_sets(pos_tag):
    v = set(VOCAB)
    m_words = monosemous_words(pos_tag)
    candidate_words = v.intersection(set(m_words))

    cmp_words = []
    seen_synsets = dict()
    for word in candidate_words:
        synset = wn.synsets(word, pos=pos_tag)[0]
        competitors = [word]
        if seen_synsets.get(synset):
            continue

        for lemma in synset.lemmas():
            other_word = lemma.name()
            if other_word != word and other_word in vocab:
                competitors.append(lemma.name())
        seen_synsets[synset] = True
        if len(competitors) > 1:
            cmp_words.append(competitors)

    return cmp_words


def get_frequency_time_series(word, start, end):
    start = str(start)
    end = str(end)
    url = f"https://books.google.com/ngrams/json?content={word}&year_start={start}&year_end={end}&smoothing=0"
    resp = requests.get(url)
    if resp.ok:
        results = json.loads(resp.content)
        if len(results) > 0:
            results = np.array(results[0]['timeseries'])
            return results
        else:
            time.sleep(1)
            base_word, _ = word.split('_')
            url = f"https://books.google.com/ngrams/json?content={base_word}&year_start={start}&year_end={end}&smoothing=0"
            resp_2 = requests.get(url)
            results_2 = json.loads(resp_2.content)
            if len(results_2) > 0:
                results_2 = np.array(results_2[0]['timeseries'])
                return results_2
            else:
                return np.zeros((191, 1))
    else:
        print(resp)
        print('something went wrong during req')


def create_aligned_spaces():
    initial_space = np.load('data/sgns/1800-w.npy')
    for i in range(1810, 2000, 10):
        file_path = 'data/sgns/{}-w.npy'.format(str(i))
        new_space = np.load(file_path)

        R, _ = orthogonal_procrustes(initial_space, new_space)
        aligned = new_space @ R
        np.save('data/rotation_matrices/Aligned_{}-w.npy'.format(str(i)), aligned)
        np.save('data/rotation_matrices/R_{}-w.npy'.format(str(i)), aligned)
        del aligned
        gc.collect()
        del new_space
        gc.collect()


def create_dataset():
    with open('data/dataset/binary_set/words_binary.npy', 'rb') as file:
        words = np.load(file)

    initial_space = np.load('data/sgns/1800-w.npy')
    X = []
    Y = []
    counter = 1
    for word in words:
        word_index = VOCAB.index(word)
        embedding = initial_space[word_index]
        _, k_nearest = k_nearest_vectors(initial_space, embedding, 4)
        init_meaning_vector = centroid(k_nearest[1:])
        distances = []
        for year in range(1810, 2000, 10):
            aligned_space = np.load('data/rotation_matrices/Aligned_{}-w.npy'.format(str(year)))
            embedding_in_other = aligned_space[word_index]
            distance = np.linalg.norm(init_meaning_vector - embedding_in_other)
            distances.append(distance)

        distances = np.array(distances)
        X.append(distances)
        Y.append(word)
        print(counter)
        counter += 1

    X = np.array(X)

    with open('data/dataset/binary_set/X_sem_dist_binary.npy', "wb") as f:
        np.save(f, X)

    with open('data/dataset/binary_set/Y_sem_dist_binary.npy', 'wb') as f:
        pkl.dump(Y, f, pkl.HIGHEST_PROTOCOL)


def load_dataset():
    X = np.load('data/dataset/X_sem_dist.npy')
    with open('data/dataset/Y_sem_dist.pkl', 'rb') as f:
        Y = pkl.load(f)
    return X, Y


def generate_interpolated_data(data):
    X_interpolated = []
    for i in range(data.shape[0]):
        y = data[i, ::, ::].reshape(1, -1).T
        x = np.arange(y.shape[1])
        cs = CubicSpline(x, y)
        x_new = np.arange(200)
        y_new = np.array(cs(x_new)).reshape(-1, 1)
        X_interpolated.append(y_new)

    np.save('data/dataset/X_interpolated.npy', np.array(X_interpolated))


def create_frequency_time_series():
    with open('data/dataset/binary_set/Y_sem_dist_binary.npy', 'rb') as file:
        words = pkl.load(file)
    pos_tags = {}
    for w in np.load('data/dataset/binary_set/words_pos_binary.npy'):
        word, pos_tag = w.split('_')
        pos_tags[word] = pos_tag

    count = 1
    X_freq = []

    for word in words:
        freq_series = get_frequency_time_series(word + '_' + pos_tags[word], 1800, 1990).flatten()

        X_freq.append(freq_series.flatten())
        print(count)
        count += 1
        time.sleep(1.5)

    X_freq = np.array(X_freq)
    np.save('data/dataset/binary_set/X_freqs_binary.npy', X_freq)


def extract_features():
    competitive_nouns = np.load("data/dataset/binary_set/cmpset_noun_binary.npy")
    competitive_adj = np.load("data/dataset/binary_set/cmpset_adj_binary.npy")
    competitive_verbs = np.load("data/dataset/binary_set/cmpset_verb_binary.npy")

    X_freq = np.load('data/dataset/binary_set/X_freqs_binary.npy')
    X_sem_dist = np.load("data/dataset/binary_set/X_sem_dist_binary.npy")
    with open("data/dataset/binary_set/Y_sem_dist_binary.npy", 'rb') as f:
        Y_sem_dist = pkl.load(f)

    X_freq_avg = []
    for i in X_freq:
        average_10 = np.average(np.array(np.split(i[:-1], 19)), axis=1)
        X_freq_avg.append(average_10)

    X_freq = np.array(X_freq_avg)

    X = []
    Y = []

    for pairs in competitive_adj:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]

        features_dist = np.concatenate((p0_sem_dist, p1_sem_dist))
        features_freq = np.concatenate((p0_freq, p1_freq))
        features = np.concatenate((features_dist, features_freq))

        X.append(features)
        Y.append(pairs[0] + '_' + pairs[1])

    for pairs in competitive_nouns:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]

        features_dist = np.concatenate((p0_sem_dist, p1_sem_dist))
        features_freq = np.concatenate((p0_freq, p1_freq))
        features = np.concatenate((features_dist, features_freq))

        X.append(features)
        Y.append(pairs[0] + '_' + pairs[1])

    for pairs in competitive_verbs:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]

        features_dist = np.concatenate((p0_sem_dist, p1_sem_dist))
        features_freq = np.concatenate((p0_freq, p1_freq))
        features = np.concatenate((features_dist, features_freq))

        X.append(features)
        Y.append(pairs[0] + '_' + pairs[1])

    X = np.array(X)
    headers = ['F1_' + str(i) for i in range(1, 20)] + \
              ['S1_' + str(i) for i in range(1, 20)] + \
              ['F2_' + str(i) for i in range(1, 20)] + \
              ['S2_' + str(i) for i in range(1, 20)]

    # Convert the NumPy array to a pandas DataFrame
    df = pd.DataFrame(X, columns=headers)
    df.to_csv('data/dataset/binary_set/X_change.csv', index=False)
    with open('data/dataset/binary_set/Y_pairs.pkl', 'wb') as f:
        pkl.dump(Y, f, pkl.HIGHEST_PROTOCOL)


def extract_change_sign():
    competitive_nouns = np.load("data/dataset/binary_set/cmpset_noun_binary.npy")
    competitive_adj = np.load("data/dataset/binary_set/cmpset_adj_binary.npy")
    competitive_verbs = np.load("data/dataset/binary_set/cmpset_verb_binary.npy")

    X_freq = np.load('data/dataset/binary_set/X_freqs_binary.npy')
    X_sem_dist = np.load("data/dataset/binary_set/X_sem_dist_binary.npy")
    with open("data/dataset/binary_set/Y_sem_dist_binary.npy", 'rb') as f:
        Y_sem_dist = pkl.load(f)

    X_freq_avg = []
    for i in X_freq:
        average_10 = np.average(np.array(np.split(i[:-1], 19)), axis=1)
        X_freq_avg.append(average_10)

    X_freq = np.array(X_freq_avg)

    X = []
    Y = []

    for pairs in competitive_adj:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]
        p0_sem_diff = np.diff(p0_sem_dist)
        p0_freq_diff = np.diff(p0_freq)
        p0_sem_change = np.sign(p0_sem_diff)[..., np.newaxis]
        p0_freq_change = np.sign(p0_freq_diff)[..., np.newaxis]
        p0_sem_freq = np.concatenate((p0_freq_change, p0_sem_change), axis=1)

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]
        p1_sem_diff = np.diff(p1_sem_dist)
        p1_freq_diff = np.diff(p1_freq)
        p1_sem_change = np.sign(p1_sem_diff)[..., np.newaxis]
        p1_freq_change = np.sign(p1_freq_diff)[..., np.newaxis]
        p1_sem_freq = np.concatenate((p1_freq_change, p1_sem_change), axis=1)

        p0_p1_change = np.concatenate((p0_sem_freq, p1_sem_freq), axis=1)

        X.append(p0_p1_change)
        Y.append(pairs[0] + '_' + pairs[1])

    for pairs in competitive_nouns:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]
        p0_sem_diff = np.diff(p0_sem_dist)
        p0_freq_diff = np.diff(p0_freq)
        p0_sem_change = np.sign(p0_sem_diff)[..., np.newaxis]
        p0_freq_change = np.sign(p0_freq_diff)[..., np.newaxis]
        p0_sem_freq = np.concatenate((p0_freq_change, p0_sem_change), axis=1)

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]
        p1_sem_diff = np.diff(p1_sem_dist)
        p1_freq_diff = np.diff(p1_freq)
        p1_sem_change = np.sign(p1_sem_diff)[..., np.newaxis]
        p1_freq_change = np.sign(p1_freq_diff)[..., np.newaxis]
        p1_sem_freq = np.concatenate((p1_freq_change, p1_sem_change), axis=1)

        p0_p1_change = np.concatenate((p0_sem_freq, p1_sem_freq), axis=1)

        X.append(p0_p1_change)
        Y.append(pairs[0] + '_' + pairs[1])

    for pairs in competitive_verbs:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]
        p0_sem_diff = np.diff(p0_sem_dist)
        p0_freq_diff = np.diff(p0_freq)
        p0_sem_change = np.sign(p0_sem_diff)[..., np.newaxis]
        p0_freq_change = np.sign(p0_freq_diff)[..., np.newaxis]
        p0_sem_freq = np.concatenate((p0_freq_change, p0_sem_change), axis=1)

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]
        p1_sem_diff = np.diff(p1_sem_dist)
        p1_freq_diff = np.diff(p1_freq)
        p1_sem_change = np.sign(p1_sem_diff)[..., np.newaxis]
        p1_freq_change = np.sign(p1_freq_diff)[..., np.newaxis]
        p1_sem_freq = np.concatenate((p1_freq_change, p1_sem_change), axis=1)

        p0_p1_change = np.concatenate((p0_sem_freq, p1_sem_freq), axis=1)

        X.append(p0_p1_change)
        Y.append(pairs[0] + '_' + pairs[1])

    X = np.array(X).reshape((-1, 4)).astype(int)
    headers = ['F_1', 'S_1', 'F_2', 'S_2']

    df = pd.DataFrame(X, columns=headers)

    df.to_csv('data/dataset/binary_set/X_change_sign.csv', index=False)


def binary_competitors():
    competitive_nouns = np.load("data/dataset/cmpset_noun.npy", allow_pickle=True)
    competitive_adj = np.load("data/dataset/cmpset_adj.npy", allow_pickle=True)
    competitive_verbs = np.load("data/dataset/cmpset_verb.npy", allow_pickle=True)

    competitive_nouns_binary = list(filter(lambda x: len(x) == 2, competitive_nouns))
    competitive_adj_binary = list(filter(lambda x: len(x) == 2, competitive_adj))
    competitive_verb_binary = list(filter(lambda x: len(x) == 2, competitive_verbs))
    words_noun = []
    for i in competitive_nouns_binary:
        words_noun.extend(i)

    words_adj = []
    for i in competitive_adj_binary:
        words_adj.extend(i)

    words_verb = []
    for i in competitive_verb_binary:
        words_verb.extend(i)

    s_n = set(words_noun)
    s_a = set(words_adj)
    s_v = set(words_verb)

    i_a_n = s_a.intersection(s_n)
    i_a_v = s_a.intersection(s_v)
    i_n_v = s_n.intersection(s_v)
    i_t = i_a_n | i_a_v | i_n_v

    final_cnb = []
    final_cab = []
    final_cvb = []

    for i in i_a_n:
        s_n.discard(i)
        s_a.discard(i)

    for i in i_a_v:
        s_v.discard(i)
        s_a.discard(i)

    for i in i_n_v:
        s_v.discard(i)
        s_n.discard(i)

    for j in competitive_nouns_binary:
        if j[0] not in i_t and j[1] not in i_t:
            final_cnb.append(j)
        else:
            if j[0] not in i_t:
                s_n.discard(j[0])
            else:
                s_n.discard(j[1])

    for j in competitive_adj_binary:
        if j[0] not in i_t and j[1] not in i_t:
            final_cab.append(j)
        else:
            if j[0] not in i_t:
                s_a.discard(j[0])
            else:
                s_a.discard(j[1])

    for j in competitive_verb_binary:
        if j[0] not in i_t and j[1] not in i_t:
            final_cvb.append(j)
        else:
            if j[0] not in i_t:
                s_v.discard(j[0])
            else:
                s_v.discard(j[1])

    s = s_a | s_n | s_v

    words_n = list(map(lambda x: x + '_NOUN', s_n))
    words_a = list(map(lambda x: x + '_ADJ', s_a))
    words_v = list(map(lambda x: x + '_VERB', s_v))
    words_postag = words_a + words_n + words_v
    np.save('data/dataset/binary_set/cmpset_adj_binary.npy', final_cab)
    np.save('data/dataset/binary_set/cmpset_noun_binary.npy', final_cnb)
    np.save('data/dataset/binary_set/cmpset_verb_binary.npy', final_cvb)
    np.save('data/dataset/binary_set/words_pos_binary.npy', words_postag)
    np.save('data/dataset/binary_set/words_binary.npy', list(s))
    print('asd')


def extract_change_numerical():
    competitive_nouns = np.load("data/dataset/binary_set/cmpset_noun_binary.npy")
    competitive_adj = np.load("data/dataset/binary_set/cmpset_adj_binary.npy")
    competitive_verbs = np.load("data/dataset/binary_set/cmpset_verb_binary.npy")

    X_freq = np.load('data/dataset/binary_set/X_freqs_binary.npy')
    X_sem_dist = np.load("data/dataset/binary_set/X_sem_dist_binary.npy")
    with open("data/dataset/binary_set/Y_sem_dist_binary.npy", 'rb') as f:
        Y_sem_dist = pkl.load(f)

    X_freq_avg = []
    for i in X_freq:
        average_10 = np.average(np.array(np.split(i[:-1], 19)), axis=1)
        X_freq_avg.append(average_10)

    X_freq = np.array(X_freq_avg)

    X = []
    Y = []

    for pairs in competitive_adj:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]
        p0_sem_diff = np.diff(p0_sem_dist)[..., np.newaxis]/10
        p0_freq_diff = np.diff(p0_freq)[..., np.newaxis]/10
        p0_sem_freq = np.concatenate((p0_sem_diff, p0_freq_diff), axis=1)

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]
        p1_sem_diff = np.diff(p1_sem_dist)[..., np.newaxis]/10
        p1_freq_diff = np.diff(p1_freq)[..., np.newaxis]/10
        p1_sem_freq = np.concatenate((p1_sem_diff, p1_freq_diff), axis=1)

        p0_p1_change = np.concatenate((p0_sem_freq, p1_sem_freq), axis=1)

        X.append(p0_p1_change)
        Y.append(pairs[0] + '_' + pairs[1])

    for pairs in competitive_nouns:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]
        p0_sem_diff = np.diff(p0_sem_dist)[..., np.newaxis]/10
        p0_freq_diff = np.diff(p0_freq)[..., np.newaxis]/10
        p0_sem_freq = np.concatenate((p0_sem_diff, p0_freq_diff), axis=1)

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]
        p1_sem_diff = np.diff(p1_sem_dist)[..., np.newaxis]/10
        p1_freq_diff = np.diff(p1_freq)[..., np.newaxis]/10
        p1_sem_freq = np.concatenate((p1_sem_diff, p1_freq_diff), axis=1)

        p0_p1_change = np.concatenate((p0_sem_freq, p1_sem_freq), axis=1)

        X.append(p0_p1_change)
        Y.append(pairs[0] + '_' + pairs[1])

    for pairs in competitive_verbs:
        p0_index = Y_sem_dist.index(pairs[0])
        p1_index = Y_sem_dist.index(pairs[1])

        p0_sem_dist = X_sem_dist[p0_index]
        p0_freq = X_freq[p0_index]
        p0_sem_diff = np.diff(p0_sem_dist)[..., np.newaxis]/10
        p0_freq_diff = np.diff(p0_freq)[..., np.newaxis]/10
        p0_sem_freq = np.concatenate((p0_sem_diff, p0_freq_diff), axis=1)

        p1_sem_dist = X_sem_dist[p1_index]
        p1_freq = X_freq[p1_index]
        p1_sem_diff = np.diff(p1_sem_dist)[..., np.newaxis]/10
        p1_freq_diff = np.diff(p1_freq)[..., np.newaxis]/10
        p1_sem_freq = np.concatenate((p1_sem_diff, p1_freq_diff), axis=1)

        p0_p1_change = np.concatenate((p0_sem_freq, p1_sem_freq), axis=1)

        X.append(p0_p1_change)
        Y.append(pairs[0] + '_' + pairs[1])

    X = np.array(X).reshape((-1, 4))
    headers = ['S_1', 'F_1', 'S_2', 'F_2']

    df = pd.DataFrame(X, columns=headers)

    df.to_csv('data/dataset/binary_set/X_change_numerical.csv', index=False)
