import random

import numpy as np


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

    @property
    def vector(self):
        vec = np.zeros((self._tfidf.shape[0], 1), dtype=np.int32)
        for i in range(len(self.clusters)):
            for m in self.clusters[i].members:
                vec[m, 0] = i + 1
        return vec

    def _recalc_centroids(self):
        for cluster in self.clusters:
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
        for d in range(self._tfidf.shape[0]):
            best_sim = 0
            best_cluster = None
            current_cluster = None
            for cluster in self.clusters:
                sim = cosine_sim(cluster.centroid, self._tfidf[d])
                if d in cluster.members:
                    current_cluster = cluster
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster
            if current_cluster != best_cluster:
                best_cluster.members.add(d)
                if current_cluster:
                    current_cluster.members.remove(d)
                self.converge = False

    def do_magic(self):
        self.__cluster()
        for i in range(self.i):
            self._recalc_centroids()
            self.__cluster()
            if self.converge:
                break


def cosine_sim(a: np.ndarray, b: np.ndarray):
    """
    :param a, b: Vector of size 1:#terms
    :return: Cosine distance
    """
    _len_a = np.sqrt(np.square(a).sum())
    _len_b = np.sqrt(np.square(b).sum())
    scalar = np.sum(np.multiply(a, b))
    return scalar / (_len_a * _len_b)


def sample_clustering():
    random_tfidf = np.random.rand(2000, 100)

    km = KMeans(tfidf=random_tfidf, k=10, max_iterations=1000, random_initial=False)
    km.do_magic()
    for cluster in km.clusters:
        print(str(cluster))
    return km.vector
