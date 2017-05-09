from sklearn.datasets import fetch_20newsgroups
from tfidf import TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import normalize
import numpy as np
import util
from scipy.sparse import csr_matrix, bmat
from typing import List
import tfidf
import os
import _pickle as pkl

StringList = List[str]

util.log("Loading data set...")
newsgroup_data = fetch_20newsgroups(remove=('headers', 'footers'))
# print(data_set.target.shape)  # categories per document
# print(data_set.filenames.shape)  # filenames per document

for doc in newsgroup_data.data:
    if len(doc) < 5:
        newsgroup_data.data.remove(doc)


class Query:
    """
    Represents a user's input query
    """

    def __init__(self, text: str):
        self.text = text.lower()
        self.tokens = tfidf.tokenize(self.text)


class TfIdfMatrix:
    """
    Functionality to compute and store tf idf representations of documents and queries
    """

    @classmethod
    def from_data_set(cls, data_set: StringList):
        """
        Create a TfIdfMatrix based on a data set
        :param data_set: list of documents as strings
        :return:
        """
        return cls(data_set)

    def __init__(self, data_set: StringList):
        self._data_set = data_set
        self._vectorizer = TfidfVectorizer()
        # filter stop words in more than 70% of documents, and filter unique words per query
        # apply L2 normalization (i.e. row*row sums up to 1)

        util.log("Calculating corpus TF-IDF...")
        self._tfidf_matrix_data_set = self._vectorizer.fit_transform(self._data_set)
        # shape n_docs x n_terms

        util.log("Calculating pairwise cosine similarity")
        self._tfidf_doc_similarity = self._tfidf_matrix_data_set.dot(self._tfidf_matrix_data_set.T)
        # shape n_docs x n_docs

        util.log("Finished.")
        self._terms = np.asarray(self._vectorizer.get_feature_names())
        # shape 1 x n_terms

        # print(self._terms)
        # print(self._tfidf_matrix_data_set.A)
        # print(self._tfidf_cosine_similarity.A)

    def get_vector_for_query(self, query: Query) -> np.ndarray:
        """
        Transform a user query into it's normalized TF-IDF-vector representation
        :param query: user query
        :return: ndarray with shape 1 x n_terms
        """
        return self._vectorizer.transform([query.text])

    def _map_term_to_column_index(self, search_term: str) -> int:
        """
        Return the column index of a search term in the feature/term list
        :param search_term:
        :return: The column-wise index of the search term in the feature list, or -1 if not found
        """
        idx = np.where(self._terms == search_term)
        found = len(idx[0])
        return idx[0][0] if found else -1

    def map_terms_to_columns(self, search_terms: StringList) -> np.ndarray:
        """
        Maps all input terms to their respective column in the matrix
        :param search_terms: List of terms
        :return: nparray containing the column indices of the found terms, shape 1 x n_found_terms
        """
        all_column_idxs = []
        for term in search_terms:
            current_column_idx = self._map_term_to_column_index(term)
            if current_column_idx is not -1:
                all_column_idxs.append(current_column_idx)

        return np.asarray(all_column_idxs)

    def get_data(self):
        """
        Get the TF IDF scores of all documents
        :return:
        """
        return self._tfidf_matrix_data_set

    def get_doc_similarities(self):
        """
        Get the pairwise similarity between all documents
        :return:
        """
        return self._tfidf_doc_similarity

    def get_number_of_documents(self):
        return self._tfidf_matrix_data_set.shape[0]

    def get_number_of_terms(self):
        return self._tfidf_matrix_data_set.shape[1]

    def get_doc_similarity_with_query(self, query: Query, relevant_doc_id: np.ndarray) -> np.ndarray:
        """
        Get the cosine similarities between all relevant documents and the query
        :param query: 
        :param relevant_doc_id: 
        :return: 
        """
        _query_tfidf = self.get_vector_for_query(query)
        _query_tfidf = normalize(_query_tfidf, norm='l2',axis=1)
        _relevant_doc_rows = self._tfidf_matrix_data_set[relevant_doc_id,:]

        return _relevant_doc_rows.dot(_query_tfidf.T)




class InvertedIndex:
    @classmethod
    def from_tf_idf_matrix(cls, tf_idf_matrix: TfIdfMatrix):
        return cls(tf_idf_matrix)

    def __init__(self, tf_idf_matrix: TfIdfMatrix):
        self._tf_idf_matrix = tf_idf_matrix

    def get_relevant_doc_ids_for_query(self, query: Query) -> np.ndarray:
        """
        Get the ids for documents containing at least one word of the query (i.e. posting lists merged with OR)
        :param query:
        :return:
        """
        column_idxs = self._tf_idf_matrix.map_terms_to_columns(query.tokens)

        tfidf_m = self._tf_idf_matrix.get_data()

        tf_columns = tfidf_m[:, column_idxs]
        # each column now contains the TF-IDF score for one term in our query

        tf_columns_summed = tf_columns.sum(axis=1)
        # sum TF-IDF row-wise. If any document contains at least one of the terms, it's row will be > 0

        combined_posting_list = np.nonzero(tf_columns_summed)[0]
        # return rows which are greater than zero

        return combined_posting_list


class AdjacencyMatrix:
    """
    Represents a (undirected) graph with "hyperlinks" between documents in form of a n_doc x n_doc (symmetric) matrix
    """

    @classmethod
    def from_cluster_and_tf_idf_matrix(cls, doc_cluster_labels: np.ndarray, tf_idf_matrix: TfIdfMatrix):
        """
        Generate a adjacency matrix based on the document's labels and their tf idf similarities
        :param doc_cluster_labels: ndarray containing the labels per document, shape 1 x n_docs
        :param tf_idf_matrix: TfIdfMatrix representation
        :return:
        """
        return cls(doc_cluster_labels, tf_idf_matrix)

    def __init__(self, doc_cluster_labels: np.ndarray, tf_idf_matrix: TfIdfMatrix):
        self._doc_cluster_labels = doc_cluster_labels
        self._tf_idf_doc_similarities = tf_idf_matrix.get_doc_similarities()

        self._cluster_sizes = np.bincount(doc_cluster_labels)
        self._rearrange = np.argsort(doc_cluster_labels)
        self._undo_rearrange = np.argsort(self._rearrange)

        util.log("Rearranging doc similarities...")
        self._sorted_doc_similarities = self._tf_idf_doc_similarities[self._rearrange, :][:, self._rearrange]
        # now the rows and columns are sorted according to their document's cluster
        # this means that the first rows and columns contain similarities
        # of documents of the first cluster with the rest etc...

        util.log("Building adjacency matrix...")
        self._adjacency_matrix = None
        self._create_graph()

        util.log("Undoing rearrangement...")
        self._adjacency_matrix = self._adjacency_matrix[self._undo_rearrange, :][:, self._undo_rearrange]
        util.log("Finished")

    def _keep_only_max_value_in_array(self, a: int, b: int, c: int, d:int) -> csr_matrix:
        """
        Returns a new sparse matrix with the same size as b-a, d-c which has zeroes except of the highest value
        :param a:
        :param b:
        :param c:
        :param d:
        :return:
        """
        sub = self._sorted_doc_similarities[a:b, c:d]
        z = np.unravel_index(sub.argmax(), sub.shape)
        i1 = z[0]
        j1 = z[1]
        new = csr_matrix(([sub[i1, j1]], ([i1], [j1])), shape=(b - a, d - c))
        return new

    def _keep_all_values_but_diag(self, a: int, b: int, c: int, d:int) -> csr_matrix:
        """
        Returns a new sparse matrix with the same sizes as b-a, d-c which retains all values, but the diagonal is set to zero
        :param a:
        :param b:
        :param c:
        :param d:
        :return:
        """
        sub = self._sorted_doc_similarities[a:b, c:d] # .tolil()
        sub.setdiag(0)
        return sub

    def _create_graph(self):
        """
        Creates the adjacency matrix which
        1. contains ALL links between documents in the same cluster
        2. contains ONLY links between the most similar values of documents in different clusters
        :return:
        """
        slices = np.insert(self._cluster_sizes, 0, 0)
        # slices is the count of all cluster's sizes, with a 0 at the start
        sizes_sum = np.cumsum(slices)
        # contains the "boundaries" index-wise, starting with 0, size_cluster1, size_cluster1 + size_cluster_2 , ...

        # acquire 2D array of size n_clusters x n_clusters to store our "chessboard" blocks
        n = len(slices)
        range_n = range(n - 1)
        blocks = [[0 for x in range_n] for y in range_n]

        for i in range(1, n):
            # loop through clusters
            current_i_min = sizes_sum[i - 1]
            current_i_max = sizes_sum[i]

            for j in range(i, n):
                # loop through clusters (only one half of the matrix since it is symmetrical)
                current_j_min = sizes_sum[j - 1]
                current_j_max = sizes_sum[j]

                if i == j:
                    # our current chessboard block needs to keep all values because we are looking at similarities
                    # of documents in the same cluster
                    # keep the blocks at the diagonal completely
                    sub_full = self._keep_all_values_but_diag(current_i_min, current_i_max, current_j_min, current_j_max)
                    blocks[i - 1][j - 1] = sub_full
                else:
                    # the blocks not on the diagonal only keep their maximum value
                    # (similarities between different clusters)
                    current_j_min = sizes_sum[j - 1]
                    current_j_max = sizes_sum[j]

                    # we can leverage the matrix symmetry and only calculate one new matrix.
                    m1 = self._keep_only_max_value_in_array(current_i_min, current_i_max, current_j_min, current_j_max)
                    m2 = m1.T

                    blocks[i - 1][j - 1] = m1
                    blocks[j - 1][i - 1] = m2

        self._adjacency_matrix = bmat(blocks).tocsr()

    def get_matrix(self):
        """
        Get the calculated adjacency matrix
        :return:
        """
        return self._adjacency_matrix


if __name__ == '__main__':
    q = Query("who likes atheism")

    x = TfIdfMatrix.from_data_set(newsgroup_data.data)

    # with open('TfIdfMatrix.pkl', 'wb') as f:
    #    pkl.dump(x, f, -1)

    i = InvertedIndex.from_tf_idf_matrix(x)

    # random clustering
    r = np.random.randint(0, 10, (x.get_number_of_documents(),))

    a = AdjacencyMatrix.from_cluster_and_tf_idf_matrix(r, x)

    # with open('AdjacencyMatrix.pkl', 'wb') as f:
    #    pkl.dump(a, f, -1)

    rel = i.get_relevant_doc_ids_for_query(q)

    print(x.get_doc_similarity_with_query(q, rel))
    # https://repl.it/HWGE/1
