from search_modules import Query, AdjacencyMatrix, TfIdfMatrix, InvertedIndex
from pagerank import PageRank
from kmeans import KMeans

import numpy as np
import util
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

from sklearn.preprocessing import normalize


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

        self.tfidf_matrix = TfIdfMatrix.from_data_set(self.newsgroup_data.data)

        self.inverted_index = InvertedIndex.from_tf_idf_matrix(self.tfidf_matrix)

        util.log("Clustering...")
        self.kmeans = KMeans(tfidf=self.tfidf_matrix.get_matrix(), k=20, max_iterations=20, random_initial=False)

        util.log("Finished.")

        r = self.kmeans.vector.ravel()

        u = np.unique(self.kmeans.vector)
        print(u)

        self.adjacency_matrix = AdjacencyMatrix.from_cluster_and_tf_idf_matrix(r, self.tfidf_matrix)

        util.log("Calculating PR...")
        pr = PageRank(adjacency_matrix=self.adjacency_matrix.get_matrix(), alpha=0.15, converge=0.01)

        util.log("Finished PR")
        pr.store_rank_vector('pr.pkl')

        self.pr_vector = pr.get_pagerank(normalized=True)

    def run_query(self, q: Query, alpha_pr: float=0.2):
        docs = self.inverted_index.get_relevant_doc_ids_for_query(q)

        pr_rel = self.pr_vector[:,docs].reshape(1,-1)
        cos_rel = self.tfidf_matrix.get_doc_similarity_with_query(q, docs).todense().reshape(1,-1)

        pr_rel_n = normalize(pr_rel,norm='max',axis=1)
        cos_rel_n = normalize(cos_rel,norm='max',axis=1)

        doc_rel = self.newsgroup_frame.iloc[docs]

        doc_rel['pr_score'] = pr_rel_n.T
        doc_rel['cos_score'] = cos_rel_n.T
        doc_rel['score'] = doc_rel.pr_score * alpha_pr + doc_rel.cos_score * (1 - alpha_pr)

        doc_rel = doc_rel.sort_values(by='score',axis=0, ascending=False)
        print(doc_rel)
        return doc_rel

if __name__ == '__main__':
    pd.options.display.max_colwidth = 100

    se = SearchEngine()

    se.run_query(Query("atheism"))
