from search_modules import Query
from search_engine import SearchEngine
from pprint import pprint

SEARCH_ENGINE = SearchEngine()

QUERIES = [
    "satellite launch",
    "national rifle association",
    "diabetes risk"
]

RELEVANT_DOCIDS = {
    "satellite launch": set([0]),
    "national rifle association": set([0]),
    "diabetes risk": set([0])
}

P_AT_N = 10
MODEL_ALPHAS = [
    0.0,
    1/3,
    2/3,
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
    return tp/len(retrieved)

def compute_p_at_k(k: int, relevant: set, retrieved: list):
    return compute_precision(relevant, set(retrieved[:k]))

def compute_rprecision(relevant: set, retrieved: list):
    return compute_p_at_k(len(relevant), relevant, retrieved)

def average_precision(relevant: set, retrieved: list):
    _sum_precisions = 0
    for rel in relevant:
        _sum_precisions += compute_p_at_k(retrieved.index(rel), relevant, retrieved)
    return _sum_precisions/len(relevant)


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
                'r-precision': compute_rprecision(RELEVANT_DOCIDS[query], ranking)
            }
    print('Document ids considered relevant by query')
    pprint(union_of_retrieved_docs)
    pprint(evaluation)
