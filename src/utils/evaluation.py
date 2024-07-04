import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm.rich import tqdm

from src.utils.dataloader import LongEvalLoader
from src.utils.eval_utils import evaluate_per_query
from src.utils.paths import loaders

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ranking',
                            type=str, default='rank.tsv', help='ranking file (qid, blank, did, rank, score, lib)')
    arg_parser.add_argument('--year', type=int, default=2022, help='year of test')
    arg_parser.add_argument('--term', type=str, default='short', help='short/long')
    arg_parser.add_argument('--dtype', type=str, default='test', help='train/test')
    arg_parser.add_argument('--output', type=str, default='output.jsonl', help='output file')
    arg_parser.add_argument('--clean', type=bool, default=True, help='clean text')
    arg_parser.add_argument('--sep', type=str, default=' ', help='separator')
    args = arg_parser.parse_args()

    ranking_path = Path(args.ranking)
    year = args.year
    term = args.term
    dtype = args.dtype
    output_path = Path(args.output)
    total_save_path = output_path.with_name(f"total_{output_path.name}")

    print(f"loading ranking from: {ranking_path}")
    print(f"saving output to: {output_path}")
    print(f"saving total output to: {total_save_path}")

    tmp_loader = loaders[str(year)][dtype]
    loader: LongEvalLoader = tmp_loader[term] if dtype == "test" else tmp_loader

    # Load ground truth
    queries = loader.load_queries()
    if args.clean:
        documents = loader.load_clean_documents()
    else:
        documents = loader.load_documents()
    try:
        qrels = loader.load_qrels()
    except:
        qrels = None

    # Load ranking file
    rankings = defaultdict(dict)
    with ranking_path.open("r") as fOut:
        for line in fOut:
            qid, _, did, _, score, _ = line.strip().split(" ")
            rankings[qid][did] = float(score)

    if qrels:
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
            if qrels:
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
