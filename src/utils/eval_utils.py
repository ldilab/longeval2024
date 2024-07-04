import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

from tqdm import tqdm

import pytrec_eval


def evaluate_per_query(qrels: Dict[str, Dict[str, int]],
                       results: Dict[str, Dict[str, float]],
                       k_values: List[int] = None,
                       ignore_identical_ids: bool = True) -> Tuple[Any, Any, Any, Any, Any]:
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 50, 100]

    if ignore_identical_ids:
        print(
            'For evaluation, we ignore identical query and document ids (default), please explicitly set '
            '``ignore_identical_ids=False`` to ignore this.'
        )
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    #!@ custom
    per_query_results = {query_id: {} for query_id in scores.keys()}

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

            #!@ custom
            per_query_results[query_id][f"NDCG@{k}"] = scores[query_id]["ndcg_cut_" + str(k)]
            per_query_results[query_id][f"MAP@{k}"] = scores[query_id]["map_cut_" + str(k)]
            per_query_results[query_id][f"Recall@{k}"] = scores[query_id]["recall_" + str(k)]
            per_query_results[query_id][f"P@{k}"] = scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        #!@ custom
        per_query_results[query_id][f"NDCG@{k}"] = round(per_query_results[query_id][f"NDCG@{k}"], 5)
        per_query_results[query_id][f"MAP@{k}"] = round(per_query_results[query_id][f"MAP@{k}"], 5)
        per_query_results[query_id][f"Recall@{k}"] = round(per_query_results[query_id][f"Recall@{k}"], 5)
        per_query_results[query_id][f"P@{k}"] = round(per_query_results[query_id][f"P@{k}"], 5)

    for eval in [ndcg, _map, recall, precision]:
        print("\n")
        for k in eval.keys():
            print("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision, per_query_results
