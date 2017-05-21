import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy import sparse
from search_modules import AdjacencyMatrix
from pprint import pprint

class PageRank:
    """
    Calculates the PageRank scores for the documents contained in the Adjacency Matrix
    =====
    adjacency_matrix = sparse matrix containing the weighted links between documents
    alpha = propapility for random teleports instead of following only links
    converge = threshold for the convergence of the PageRank algorithm
    """
    def __init__(self, adjacency_matrix: AdjacencyMatrix, alpha: float, converge: float):
        self._matrix = adjacency_matrix
        # self._make_sparse_google_matrix(alpha)
        self._make_google_matrix(alpha)
        self._calculate_rank(converge)

    def _make_stochastic(self):
        """
        Fills rows only containing zeros with equal probabilities
        0 -> 1/N
        """
        n = self._matrix.shape[0]
        #  boolean array with 1 for rows that are zero
        row_zeros = np.transpose([np.array(self._matrix.sum(axis=1))[:, 0] == 0])
        row_filler = np.array([np.ones(n, dtype=np.float)])
        a = row_zeros * row_filler
        S = self._matrix + a * (1/n * np.array([np.ones(n, np.float)]))
        self._matrix = S

    def _make_google_matrix(self, alpha):
        """
        Creates the full Google Matrix used to calculate PageRank on Matrices
        """
        self._make_stochastic()
        e = np.array([np.ones(self._matrix.shape[0])])
        G = alpha * self._matrix + (1 - alpha) * 1 / self._matrix.shape[0] * e * np.transpose(e)
        self._matrix = G
        pprint(G)

    def _make_sparse_google_matrix(self, alpha):
        """
        Creates the full Google Matrix using the Sparse Matrix H
        """
        n = self._matrix.shape[0]

        # Creates 'a' which is a matrix with rows filled that have row sum 0 in H
        row_zeros = np.transpose([np.array(self._matrix.sum(axis=1))[:, 0] == 0])
        row_filler = np.array([np.ones(n, dtype=np.float)])
        a = row_zeros * row_filler

        e = np.array([np.ones(n)])
        e_t = np.transpose(e)

        G = alpha * self._matrix + (alpha * a + (1 - alpha) * e) * 1 / n * e_t
        self._matrix = G
        pprint(G)

    def _calculate_rank(self, converge=0.0001):
        """
        Calculate the PageRank for the G-Matrix
        convergence is the threshold until which we will iteratively compute the pagerank scores
        """
        # number of nodes in the graph
        n_nodes = self._matrix.shape[0]

        # Array of row counts necesssary to identify sinks in the graph
        r_sums = np.array(self._matrix.sum(1))[:, 0]
        row_index, col_index = self._matrix.nonzero()

        # Creating Row Normalized Version of our matrix
        # self._matrix.data /= r_sums[row_index]
        self._matrix.data /= np.transpose([r_sums])


        # initialize r_0 and r_1 to track changes between each state
        r_0, r_1 = np.zeros(n_nodes), np.ones(n_nodes)/ n_nodes
        # Repeat until the convergence criteria is met
        while np.sum(np.abs(r_1-r_0)) > converge:
            r_0 = r_1.copy()
            r_1 = r_0 * self._matrix
            # calculate pagerank one node at a time
        self._ranking = r_1

    def get_pagerank(self, normalized: bool=True):
        if normalized:
            return self._ranking/float(sum(self._ranking))
        else:
            return self._ranking

if __name__ == '__main__':
    H = np.array([[0,0,1,0,0,0,0],
                  [0,1,1,0,0,0,0],
                  [1,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0],
                  [1,0,1,0,0,0,1],
                  [0,1,0,0,0,1,1],
                  [0,1,0,1,1,0,1]])
    print(H)
    h_mat = bmat(H).tocsr()
    pageRank = PageRank(h_mat, 0.85, 0.0001)
    print(pageRank.get_pagerank(False))