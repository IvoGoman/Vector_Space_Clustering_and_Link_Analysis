from search_modules import Query
from search_engine import SearchEngine
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


SEARCH_ENGINE = SearchEngine()

RELEVANT_DOCIDS = {
    "national rifle association": set([455, 590, 4802, 4959, 8285, 10774]),
    "diabetes risk": set([3059, 3258, 7425, 454]),
    "NASA Apollo": set([9350,10658,7208,4649,6490,6856,7208]),
    "sound driver" : set([3167,9563,10812,8094,5767,9754,10116,2946]),
    "color printer" : set([4236,10473, 1146])
}

QUERIES = [k for k in RELEVANT_DOCIDS.keys()]

P_AT_N = 10
MODEL_ALPHAS = [
    0.0,
    0.2,
    1/3,
    0.5,
    2/3,
    0.8,
    1
]

def run_queries(queries: list, alphas: list, n: int) -> dict:
    res = {}
    for q in queries:
        res[q] = {}
        for a in alphas:
            ranks = []
            raw = SEARCH_ENGINE.run_query(Query(q), a)
            for row in raw.iterrows():
                ranks.append(
                    {'id': str(row[0]), 'text': row[1]['text'], 'score': row[1]['score']}
                )
            ranks = sorted(ranks, key=lambda x:x['score'], reverse=True)[:n]
            for i in range(len(ranks)):
                ranks[i]['rank'] = i+1
            res[q][a] = ranks
    return res


def compute_precision(relevant: set, retrieved: set):
    tp = len(set.intersection(relevant, retrieved))
    if len(retrieved) == 0:
        return 0.0
    return tp/len(retrieved)

def compute_p_at_k(k: int, relevant: set, retrieved: list):
    return compute_precision(relevant, set(retrieved[:k]))

def compute_rprecision(relevant: set, retrieved: list):
    return compute_p_at_k(len(relevant), relevant, retrieved)

def average_precision(relevant: set, retrieved: list):
    _sum_precisions = 0
    for rel in relevant.intersection(set(retrieved)):
        _sum_precisions += compute_p_at_k(retrieved.index(rel), relevant, retrieved)
    return _sum_precisions/len(relevant)

def mean_average_precision(average_precisions:list):
    return sum(average_precisions)/len(average_precisions)


def pooling(query: dict):
    """
    This builds the union from different models of a given query
    :param ranking: A dict containing the different model names as keys and as value an array of dicts that contains the key 'id' with the document id that the model considered as relevant
    :return: the union of all as relevant considered docs
    """
    retrieved = set()
    for alpha in query.values():
        for ranking in alpha:
            retrieved.add(ranking)
    return retrieved

def obtain_ranked_array(ranking_query: dict) -> dict:
    ranked_dict_per_query = {}
    for query, alphas in ranking_query.items():
        ranked_dict_per_query[query] = {}
        for alpha, ranking in alphas.items():
            ranked_dict_per_query[query][alpha] = [int(doc['id']) for doc in ranking]
    return ranked_dict_per_query


def plot(y:list, legend: list, xlabel, ylabel, filename):
    X = np.array(MODEL_ALPHAS)
    graphs = []
    for g in range(len(y)):
        graphs.append(plt.plot(X, np.array(y[g]), label=legend[g])[0])
    plt.legend(handles=graphs)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.gcf().clear()


if __name__ == '__main__':
    ranking_query = run_queries(QUERIES, MODEL_ALPHAS, P_AT_N)
    ranked_array_by_query = obtain_ranked_array(ranking_query)
    evaluation = {}
    union_of_retrieved_docs = {}
    for query, alphas in ranked_array_by_query.items():
        union_of_retrieved_docs[query] = pooling(alphas)
        evaluation[query]= {}
        for alpha, ranking in alphas.items():
            evaluation[query][alpha] = {
                'r-precision': compute_rprecision(RELEVANT_DOCIDS[query], ranking),
                'average_precision': average_precision(RELEVANT_DOCIDS[query], ranking)
            }
    print('Document ids considered relevant by query')
    pprint(union_of_retrieved_docs)
    pprint(evaluation)

    legend = QUERIES
    r_precision = []
    for q in legend:
        r_precision.append([evaluation[q][a]['r-precision'] for a in MODEL_ALPHAS])
    plot(r_precision, legend, xlabel='alpha value', ylabel='r-precision', filename='r-precision.png')

    legend = ['Mean Average Precision']
    average_precisions = []
    for alpha in MODEL_ALPHAS:
        average_precisions.append([evaluation[q][alpha]['average_precision'] for q in QUERIES])
    mean_average_precisions = [[mean_average_precision(ap) for ap in average_precisions]]
    plot(mean_average_precisions, legend, xlabel='alpha value', ylabel=legend[0], filename='mean_average_precision.png')

