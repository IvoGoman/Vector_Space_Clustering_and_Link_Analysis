from search_modules import Query, AdjacencyMatrix, TfIdfMatrix, InvertedIndex
from pagerank import PageRank
from kmeans import KMeans

import numpy as np
import util
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

from sklearn.preprocessing import normalize
import timeit

from typing import List
IntList = List[int]

class SearchEngine:
    def __init__(self):
        util.log("Loading data set...")
        self.newsgroup_data = fetch_20newsgroups(remove=('headers', 'footers'))


        # print(data_set.target.shape)  # categories per document
        # print(data_set.filenames.shape)  # filenames per document

        for doc in self.newsgroup_data.data:
            if len(doc) < 5:
                self.newsgroup_data.data.remove(doc)

        self.newsgroup_frame = pd.DataFrame.from_dict({'text': self.newsgroup_data.data})

        #f = self.newsgroup_frame.text.str.contains('National Rifle Association')

        #ids = np.arange(len(self.newsgroup_frame))[f]
        #self.list_docs(ids)
        #return

        self.tfidf_matrix = TfIdfMatrix.from_data_set(self.newsgroup_data.data)

        self.inverted_index = InvertedIndex.from_tf_idf_matrix(self.tfidf_matrix)

        util.log("Clustering...")
        self.kmeans = KMeans(tfidf=self.tfidf_matrix.get_matrix(), k=20, max_iterations=20, random_initial=False)

        try:
            self.kmeans.load_cluster_vector('cluster_vector.pkl')
        except FileNotFoundError:
            self.kmeans.do_magic()
            self.kmeans.store_cluster_vector('cluster_vector.pkl')

        util.log("Finished.")

        r = self.kmeans.vector.ravel()
        u = np.unique(self.kmeans.vector)
        print(u)

        self.adjacency_matrix = AdjacencyMatrix.from_cluster_and_tf_idf_matrix(r, self.tfidf_matrix)
        try:
            pr = PageRank(pickle='pr.pkl')
        except FileNotFoundError:
            util.log("No precomputed PageRank...")
            util.log("Calculating PR...")
            pr = PageRank(adjacency_matrix=self.adjacency_matrix.get_matrix(), alpha=0.85, converge=0.00001)
        util.log("Finished PR")
        pr.store_rank_vector('pr.pkl')

        self.pr_vector = pr.get_pagerank(normalized=True)

    def list_docs(self, ids):
        x = self.newsgroup_frame.iloc[ids]
        for idx, r in x.iterrows():
            print(idx)
            print(' '.join(r.text.split()))
            print()

    def run_query(self, q: Query, alpha_pr: float=0.2) -> pd.DataFrame:
        docs = self.inverted_index.get_relevant_doc_ids_for_query(q)

        pr_rel = self.pr_vector[:,docs].reshape(1,-1)
        cos_rel = self.tfidf_matrix.get_doc_similarity_with_query(q, docs).todense().reshape(1,-1)

        pr_rel_n = normalize(pr_rel,norm='max',axis=1)
        cos_rel_n = normalize(cos_rel,norm='max',axis=1)

        doc_rel = self.newsgroup_frame.iloc[docs].copy()

        doc_rel['pr_score'] = pr_rel_n.T
        doc_rel['cos_score'] = cos_rel_n.T
        doc_rel['score'] = doc_rel.pr_score * alpha_pr + doc_rel.cos_score * (1 - alpha_pr)

        doc_rel = doc_rel.sort_values(by='score',axis=0, ascending=False)
        # print(doc_rel)
        return doc_rel

    def p_at_r(self, q: Query, rel_ids: IntList, alpha_pr: float=0.2):
        res = self.run_query(q,alpha_pr)

        current_pos = 0
        for idx, row in res.iterrows():
            current_pos += 1
            if idx in rel_ids:
                print("Found", idx, "at", current_pos)
                rel_ids.remove(idx)
            if len(rel_ids) is 0:
                print("All found at", current_pos)
                print("P@R:", 10.0 / current_pos)



if __name__ == '__main__':
    pd.options.display.max_colwidth = 100

    se = SearchEngine()

    """se.list_docs([455, # R
                                590, # R
                                2183, # N
                                3403, # N
                                3738, # N
                                4263, # N
                                4802, # N
                                4959, # N
                                5988, # N
                                6219, # N
                                6332, # N
                                6623, # N
                                6920, # N
                                7112, # N
                                7126, # N
                                7241, # N
                                8285, # N
                                8467, # N
                                9104, # N
                                9605, # N
                                9625, # N
                                9641]) # N"""

