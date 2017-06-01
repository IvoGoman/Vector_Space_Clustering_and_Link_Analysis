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
        self.doc_cluster = [None for i in range(self._tfidf.shape[0])]
        self.converge = None
        if random_initial:
            self.__initialize_clusters_with_random_centroids()
        else:
            self.__initialize_cluster_with_random_document()

        self.__vector = None

    @property
    def vector(self):
        if self.__vector is None:
            vec = np.zeros((self._tfidf.shape[0], 1), dtype=np.int32)
            for i in range(len(self.clusters)):
                for m in self.clusters[i].members:
                    vec[m, 0] = i + 1
            return vec
        else:
            return self.__vector

    def __recalc_centroids(self):
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
        max_vals = np.argmax(similarities, axis=0)
        for doc_idx in range(max_vals.shape[1]):
            best_cluster_idx = max_vals[0, doc_idx]
            if self.doc_cluster[doc_idx] != best_cluster_idx:
                if self.doc_cluster[doc_idx] is not None:
                    old_cluster = self.doc_cluster[doc_idx]
                    self.clusters[old_cluster].members.remove(doc_idx)
                self.doc_cluster[doc_idx] = best_cluster_idx
                self.clusters[best_cluster_idx].members.add(doc_idx)
                self.converge = False

    def do_magic(self):
        for i in range(self.i):
            util.log('kmeans iteration %s starting' % str(i))
            self.__recalc_centroids()
            self.__cluster()
            if self.converge:
                break

    def store_cluster_vector(self, file):
            with open(file, 'wb') as f:
                pkl.dump(self.vector, f)

    def load_cluster_vector(self, file):
        self.__vector = pkl.load(open(file, "rb"))
