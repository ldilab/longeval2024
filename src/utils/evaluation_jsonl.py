import argparse


import pandas as pd

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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ranking',
                            type=str, default='rank.jsonl',
                            help='jsonl file with new ranking.\n'
                                 'new_rankings must be a dictionary of {docid: score} on field, "new_rankings"\n'
                                 'new_rankings must be placed for each query.')
    args = arg_parser.parse_args()

    ranking_path = Path(args.ranking)
    total_save_path = ranking_path.with_name(f"total_metric_{ranking_path.name}").with_suffix(".tsv")
    output_path = ranking_path.with_name(f"output_evaluation_{ranking_path.name}")

    print(f"loading ranking from: {ranking_path}")
    print(f"saving total output to: {total_save_path}")
    print(f"saving output to: {output_path}")

    queries = defaultdict(str)
    documents = defaultdict(dict)
    qrels = defaultdict(dict)
    rankings = defaultdict(dict)
    with ranking_path.open("r") as fOut:
        for line in fOut:
            obj = json.loads(line)
            qid = obj["qid"]
            queries[qid] = obj["text"]
            for doc in obj["top1000"]:
                did = doc["docid"]
                documents[did] = {"text": doc["text"]}

            qrels[qid] = obj["qrels"]

            rankings[qid] = {
                did: score
                for did, score in obj["new_rankings"].items()
            }

    if len(qrels) > 0:
        # Evaluate
        *results, results_per_query = evaluate_per_query(qrels, rankings, ignore_identical_ids=False)

        results_to_df = []
        for result in results:
            metric_name = list(result.keys())[0].split("@")[0]
            metric_values = result.values()
            row = [metric_name] + list(metric_values)
            results_to_df.append(row)
        results_df = (
            pd.DataFrame(results_to_df, columns=["metric", "@1", "@3", "@5", "@10", "@20", "@50", "@100"])
            .set_index("metric")
            .T
        )

        # Save total results
        results_df.to_csv(total_save_path, sep="\t", index=True, header=True, float_format="%.4f")

    # Save results
    with output_path.open("w") as fOut:
        for qid, query in tqdm(queries.items(), desc="Saving results"):
            top1000_dids = list(map(
                lambda x: x[0],
                sorted(rankings[qid].items(), key=lambda x: x[1], reverse=True)[:1000]
            ))
            top1000_save_obj = [
                {"docid": did, "bm25_score": score, "rank": rank, "text": documents[did]["text"]}
                for rank, (did, score) in enumerate(zip(top1000_dids, [rankings[qid][did] for did in top1000_dids]), 1)
            ]
            if len(qrels) > 0:
                qrels_save_obj = qrels[qid]
                bm25_results_save_obj = results_per_query[qid] if qid in results_per_query else []
            else:
                qrels_save_obj = None
                bm25_results_save_obj = None

            save_obj = {
                "qid": qid,
                "text": query,
                "top1000": top1000_save_obj,
                "qrels": qrels_save_obj,
                "bm25_results": bm25_results_save_obj
            }

            fOut.write(json.dumps(save_obj) + "\n")
