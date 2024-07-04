import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.rich import tqdm




import re
from nltk.corpus import stopwords


import nltk

from src.utils.beir_utils import GenericDataLoader

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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


if __name__ == '__main__':

    output_dir = Path("/workspace/longeval/src/output/msmarco")

    doc_words_path = output_dir / "words.jsonl"
    doc_dist_path = output_dir / "dist.json"
    doc_stat_path = output_dir / "stat.json"

    print("saving words to", doc_words_path)
    print("saving distribution to", doc_dist_path)
    print("saving stats to", doc_stat_path)

    corpus, corpusid2line = GenericDataLoader(
        data_folder="/workspace/beir/msmarco",
        rank=0
    ).load_custom(split="train", which="corpus")


    words = defaultdict(int)
    stats = defaultdict(int)
    with doc_words_path.open("w") as fOut:
        for did, document in tqdm(corpus.items()):
            doc_string = document["text"] if len(document["title"]) == 0 else document["title"] + " " + document["text"]

            doc_string = doc_string.lower()
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
