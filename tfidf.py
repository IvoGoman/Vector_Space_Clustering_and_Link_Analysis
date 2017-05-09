import numpy as np
import re
from typing import List
from collections import Counter
from scipy.sparse import coo_matrix, diags
import math
from sklearn.preprocessing import normalize

StringList = List[str]


regex_tokenize = re.compile(r'(?u)\b\w\w+\b')


np.set_printoptions(precision=2)

def tokenize(s: str) -> StringList:
    return regex_tokenize.findall(s.lower())


class TfidfVectorizer:
    def __init__(self):
        self._idf_vector = None
        self._vocabulary = None
        self._tf_idf_matrix = None
        self._df_counter = None
        self._terms = None
        self._term_position = None
        self._term_set = set()
        self._n_docs = 0
        self._n_terms = 0

    def fit_transform(self, data_set: StringList) -> np.ndarray:
        """
        Learn the vocabulary and give back the transformed matrix
        :param data_set: 
        :return: 
        """
        self.fit(data_set)
        return self._tf_idf_matrix

    def transform(self, data_set: StringList) -> np.ndarray:
        """
        Transform (queries/docs) based on already learned vocabulary and IDF vector
        :param data_set: 
        :return: 
        """

        sparse_i = []
        sparse_j = []
        sparse_data = []

        doc_id = 0

        for doc in data_set:
            tokens = tokenize(doc)

            doc_counter = Counter()
            doc_counter.update(tokens)
            # update raw term frequencies per document

            max_freq = None

            for token, frequency in doc_counter.most_common():
                if max_freq is None:
                    max_freq = 1 + math.log10(frequency)

                if token not in self._term_set:
                    continue

                sparse_i.append(doc_id)
                sparse_j.append(self._term_position[token])

                freq_log = (1 + math.log10(frequency)) / max_freq
                # apply logscale on TF

                sparse_data.append(freq_log)

            doc_id += 1

            tf_idf_matrix = coo_matrix(
                (sparse_data, (sparse_i, sparse_j)),
                shape=(doc_id, self._n_terms), dtype=np.float32)

            tf_idf_matrix = tf_idf_matrix.dot(diags(self._idf_vector))

            tf_idf_matrix = normalize(tf_idf_matrix,axis=1, norm='l2')
            return tf_idf_matrix

    def fit(self, data_set: StringList):
        """
        Learn IDF vector and vocabulary 
        :param data_set: 
        :return: 
        """

        self._n_docs = len(data_set)

        self._df_counter = Counter()
        # count overall tokens and their occurences (document frequency)

        doc_id = 0

        terms_list = []
        self._term_position = {}
        # store terms and their order

        sparse_i = []
        sparse_j = []
        sparse_data = []

        for doc in data_set:
            tokens = tokenize(doc)
            token_set = set(tokens)
            new_tokens = token_set.difference(self._term_set)

            for new_token in new_tokens:
                self._term_position[new_token] = len(self._term_position)
                terms_list.append(new_token)

            self._term_set = self._term_set.union(new_tokens)

            self._df_counter.update(token_set)
            # increase document frequency by max 1 per doc and term (use set)

            doc_counter = Counter()
            doc_counter.update(tokens)
            # update raw term frequencies per document

            max_freq = None

            for token, frequency in doc_counter.most_common():
                if max_freq is None:
                    max_freq = 1 + math.log10(frequency)

                sparse_i.append(doc_id)
                sparse_j.append(self._term_position[token])

                freq_log = (1 + math.log10(frequency)) / max_freq
                # apply logscale on TF

                sparse_data.append(freq_log)

            doc_id += 1

        self._n_terms = len(self._term_set)
        self._terms = np.asarray(terms_list)

        # construct the sparse TF matrix
        self._tf_idf_matrix = coo_matrix(
            (sparse_data, (sparse_i, sparse_j)),
            shape=(self._n_docs, self._n_terms), dtype=np.float32)

        df_vec = np.asarray([self._df_counter[t] for t in self._terms],dtype=np.float32)
        df_vec = np.multiply(np.reciprocal(df_vec), self._n_docs)
        self._idf_vector = np.log10(df_vec)
        # calc the IDF vector based on log(D / DF)

        self._tf_idf_matrix = self._tf_idf_matrix.dot(diags(self._idf_vector))
        self._tf_idf_matrix = normalize(self._tf_idf_matrix, axis=1, norm='l2')

    def get_feature_names(self) -> StringList:
        return self._terms

    def _n_docs(self):
        return self._n_docs

    def _n_terms(self):
        return self._n_terms


