import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.rich import tqdm
from transformers import T5Tokenizer


import re
from nltk.corpus import stopwords

from src.utils.dataloader import LongEvalLoader
from src.utils.paths import loaders

STOPWORDS = set(stopwords.words("english"))
from string import punctuation

PUNCTUATIONS = set(punctuation)


def regularize_tokens(tokens: List[str]):
    tokens = [t.strip() for t in tokens]
    tokens = [t.strip("".join(PUNCTUATIONS)) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [t for t in tokens if t not in PUNCTUATIONS]
    tokens = [t for t in tokens if not re.match(r"^\d+?\.\d+?$", t)]  # e.g., 1.23
    tokens = [t for t in tokens if not re.match(r"^\d+?\,\d+?$", t)]  # e.g., 1,234
    tokens = [t for t in tokens if not t.isnumeric()]  # e.g., 123
    return tokens


def filter_by_tokenizer(toknzr: T5Tokenizer, sent: str, length: int=1024):
    tokens = toknzr([sent], return_tensors='pt', max_length=length, truncation=True)
    truncated_token_ids = tokens['input_ids'][0]
    decoded_string = toknzr.decode(truncated_token_ids)
    return decoded_string


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--year', type=int, default=2022, help='year of test')
    arg_parser.add_argument('--term', type=str, default='short', help='short/long')
    arg_parser.add_argument('--dtype', type=str, default='test', help='train/test')
    arg_parser.add_argument('--which', type=str, default='document', help='query/document')
    args = arg_parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained('Soyoung97/ListT5-3b', legacy=False)

    year = args.year
    term = args.term
    dtype = args.dtype

    tmp_loader = loaders[str(year)][dtype]
    loader: LongEvalLoader = tmp_loader[term] if dtype == "test" else tmp_loader

    if args.which == "document":
        output_dir = loader.document_dir
    elif args.which == "query":
        output_dir = loader.query_dir
    else:
        raise ValueError("which should be either query or document")

    doc_words_path = output_dir / "tok_words.jsonl"
    doc_dist_path = output_dir / "tok_dist.json"
    doc_stat_path = output_dir / "tok_stat.json"

    print("saving words to", doc_words_path)
    print("saving distribution to", doc_dist_path)
    print("saving stats to", doc_stat_path)

    if args.which == "query":
        documents = loader.load_queries()
    elif args.which == "document":
        documents = loader.load_clean_documents()

    words = defaultdict(int)
    stats = defaultdict(int)
    with doc_words_path.open("w") as fOut:
        for did, document in tqdm(documents.items()):
            if args.which == "query":
                doc_string = document
            elif args.which == "document":
                doc_string = document["text"]

            doc_string = doc_string.lower()
            doc_string = filter_by_tokenizer(tokenizer, doc_string)
            doc_string = doc_string.split()
            doc_words = regularize_tokens(doc_string)

            fOut.write(json.dumps({"id": did, "words": doc_words}) + "\n")

            for word in doc_words:
                words[word] += 1

            stats[did] = len(doc_string)

    with doc_dist_path.open("w") as fOut:
        fOut.write(json.dumps(words) + "\n")

    with doc_stat_path.open("w") as fOut:
        fOut.write(json.dumps(stats) + "\n")

    print("average number of words per document:", sum(stats.values()) / len(stats))
