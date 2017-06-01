import _pickle as pkl
import os.path
import util
import numpy as np
from scipy.sparse import csr_matrix, bmat
import scipy as scipy
from sklearn.preprocessing import normalize


class PageRank:
    """
    Calculates the PageRank scores for the documents contained in the Adjacency Matrix
  
    adjacency_matrix = sparse matrix containing the weighted links between documents
    alpha = propapility for random teleports instead of following only links
    converge = threshold for the convergence of the PageRank algorithm
    """

    def __init__(self, adjacency_matrix=np.ndarray, alpha: float = None, converge: float = None, pickle=None):
        if pickle is None:
            self._matrix = adjacency_matrix
            self._alpha = alpha
            self._converge = converge
            self._max_iter = 100
            self._run()
        else:
            if not os.path.isfile(pickle):
                raise FileNotFoundError("The file does not exist.")
            self._load_rank_vector(pickle)

    def _run(self):
        """
        Calculates the PageRank Scores for the given Adjacency Matrix
        """
        n = self._matrix.shape[0]
        e_t = np.array([np.ones(n)])
        e = np.transpose(e_t)
        # initialize r_0 and r_1 to track changes between each state
        r_0, r_1 = np.zeros(n), np.ones(n) / n
        # Creates 'a' which is a matrix with rows filled that have row sum 0 in H
        a = np.transpose([np.array(self._matrix.sum(axis=1))[:, 0] == 0]).astype(np.float64)
        # Row normalize the matrix
        self._matrix = normalize(self._matrix, norm='l1', axis=1)
        i = 0
        while (np.sum(np.abs(r_1 - r_0)) > self._converge) and (i < self._max_iter):
            r_0 = r_1.copy()
            r_1 = self._alpha * csr_matrix.dot(r_0, self._matrix) + (self._alpha * np.dot(r_0, a) + (1 - self._alpha) * np.dot(r_0, e)) * (e_t / n)
            i += 1
            util.log("PageRank Iteration: " + str(i))
        self._ranking = r_1


    def get_pagerank(self, normalized: bool = True):
        if normalized:
            return self._ranking / float(np.sum(self._ranking))
        else:
            return self._ranking

    def store_rank_vector(self, pickle):
        with open(pickle, 'wb') as f:
            pkl.dump(self._ranking, f, -1)

    def _load_rank_vector(self, pickle):
        self._ranking = pkl.load(open(pickle, "rb"))


if __name__ == '__main__':
    H = np.array([[0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0]])
    print(H)
    h_mat = bmat(H).tocsr()
    pageRank = PageRank(h_mat, 0.9, 0.0001)
    print(pageRank.get_pagerank(True))

