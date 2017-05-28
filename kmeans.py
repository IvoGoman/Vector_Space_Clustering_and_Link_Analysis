import random
from scipy.sparse import csr_matrix, bmat
import numpy as np
import _pickle as pkl
import util


class Cluster:
    def __init__(self, initial_centroid: np.ndarray):
        self.centroid = initial_centroid
        self.members = set()

    def __str__(self):
        return "Centroid: %s\nMembers: %s" % (str(self.centroid), len(self.members))


class KMeans:
    def __init__(self, tfidf: np.ndarray, k: int, max_iterations: int, random_initial: bool):
        self._tfidf = tfidf
        self.k = k
        self.i = max_iterations
        self.clusters = []
        self.converge = None
        if random_initial:
            self.__initialize_clusters_with_random_centroids()
        else:
            self.__initialize_cluster_with_random_document()

        self.__vector = None

    @property
    def vector(self):
        if len(self.__vector) == 0:
            vec = np.zeros((self._tfidf.shape[0], 1), dtype=np.int32)
            for i in range(len(self.clusters)):
                for m in self.clusters[i].members:
                    vec[m, 0] = i + 1
            return vec
        else:
            return self.__vector

    def _recalc_centroids(self):
        for cluster in self.clusters:
            if len(cluster.members) != 0:
                mem = np.array([self._tfidf[m] for m in cluster.members])
                _sum = np.sum(mem, axis=0)
                cluster.centroid = _sum / float(len(cluster.members))

    def __initialize_cluster_with_random_document(self):
        rand_idx = (random.sample(range(self._tfidf.shape[0] - 1), k=self.k))
        [self.clusters.append(Cluster(initial_centroid=self._tfidf[idx])) for idx in rand_idx]

    def __initialize_clusters_with_random_centroids(self):
        centroids = np.random.rand(self.k, self._tfidf.shape[1])
        [self.clusters.append(Cluster(initial_centroid=centroid)) for centroid in centroids]

    def __cluster(self):
        self.converge = True
        centroids = bmat([[item for item in c.centroid[0]] for c in self.clusters])
        similarities = centroids.dot(self._tfidf.T)
        for d in range(similarities.shape[1]):
            best_sim = 0
            best_cluster = None
            current_cluster = None
            for c in range(similarities.shape[0]):
                sim = similarities[c,d]
                if d in self.clusters[c].members:
                    current_cluster = c
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c
            if current_cluster != best_cluster:
                self.clusters[best_cluster].members.add(d)
                if current_cluster:
                    self.clusters[current_cluster].members.remove(d)
                self.converge = False

    def do_magic(self):
        for i in range(self.i):
            util.log('kmeans iteration %s starting' % str(i))
            self._recalc_centroids()
            self.__cluster()
            if self.converge:
                break

    def store_cluster_vector(self, file):
            with open(file, 'wb') as f:
                pkl.dump(self.vector, f)

    def load_cluster_vector(self, file):
        self.__vector = pkl.load(open(file, "rb"))



def cosine_sim(a: csr_matrix, b: csr_matrix):
    """
    :param a, b: Vector of size 1:#terms
    :return: Cosine distance
    """
    _len_a = np.sqrt((a.power(2)).sum())
    _len_b = np.sqrt((b.power(2)).sum())
    scalar = float((a.multiply(b)).sum())
    return scalar / (_len_a * _len_b)

