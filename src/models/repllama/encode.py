import ast
import glob
import logging
import os
import pickle
import sys
from contextlib import nullcontext
from itertools import chain

import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset, Features, Value
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import ModelArguments, DataArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from data import HFQueryDataset, HFCorpusDataset, CustomQueryDataset, CustomCorpusDataset

from repllama import RepLLaMA
from data import EncodeDataset, EncodeCollator
from tevatron.faiss_retriever.__main__ import pickle_load
from utils import replace_with_xformers_attention

import json
from collections import defaultdict
from pathlib import Path

from tqdm.rich import tqdm

logger = logging.getLogger(__name__)


class LongEvalLoader:
    def __init__(self, data_dir: Path, query_file_name: str = None, qrels_file_name: str = "train.txt"):
        assert query_file_name is not None, "query_file_name must be provided"

        self.data_dir = data_dir
        if type(data_dir) == str:
            data_dir = Path(data_dir)
        self.document_dir = data_dir / "Documents" / "Json"
        self.clean_document_path = self.document_dir.parent / "clean_corpus.jsonl"
        self.document_paths = list(self.document_dir.glob("*.json"))

        self.query_dir = data_dir / "Queries"
        self.query_file_name = query_file_name
        self.query_path = self.query_dir / self.query_file_name

        self.qrels_dir = data_dir / "Qrels"
        self.qrels_file_name = qrels_file_name
        self.qrels_path = self.qrels_dir / self.qrels_file_name

    def load_clean_documents(self, dids=None):
        n_lines_clean_document = sum(1 for _ in self.clean_document_path.open())
        documents = {}
        target_dids = defaultdict(int)
        if dids:
            for did in tqdm(dids):
                target_dids[did] = 1
        with self.clean_document_path.open("r") as f:
            for line in tqdm(f, total=n_lines_clean_document):
                document = json.loads(line)
                doc_id = document["id"]
                if dids:
                    if target_dids[doc_id] == 1:
                        continue

                documents[doc_id] = {
                    "title": document["title"],
                    "text": document["text"],
                }

        return documents

    def load_documents(self):
        documents = {}
        for document_path in tqdm(self.document_paths):
            with open(document_path, "r") as f:
                document_lines = json.load(f)
                for document_line in document_lines:
                    doc_id = document_line["id"]
                    documents[doc_id] = {
                        "title": "",
                        "text": document_line["contents"],
                    }

        return documents

    def load_queries(self):
        queries = {}
        with open(self.query_path, "r") as f:
            # query_path is tsv file
            for line in f:
                query_id, query = line.strip().split("\t")
                queries[query_id] = query
        return queries

    def load_qrels(self):
        qrels = defaultdict(dict)
        with open(self.qrels_path, "r") as f:
            # qrels_path is tsv file
            for line in f:
                if len(line.strip()) == 0:
                    continue
                query_id, _, doc_id, score = line.strip().split(" ")
                qrels[query_id][doc_id] = int(score)

        return qrels

    def load_all(self):
        documents = self.load_documents()
        queries = self.load_queries()
        qrels = self.load_qrels()
        return documents, queries, qrels


longeval_dataset_dir = Path("/workspace/longeval/datasets")

dataset_2022_dir = longeval_dataset_dir / "2022"
dataset_2022_train_dir = dataset_2022_dir / "publish" / "English"
dataset_2022_train_heldout_loader = LongEvalLoader(dataset_2022_train_dir, "heldout.tsv")
dataset_2022_train_main_loader = LongEvalLoader(dataset_2022_train_dir, "train.tsv")

dataset_2022_test_short_dir = dataset_2022_dir / "test-collection" / "A-Short-July" / "English"
dataset_2022_test_short_loader = LongEvalLoader(dataset_2022_test_short_dir, "test07.tsv", "a-short-july.txt")

dataset_2022_test_long_dir = dataset_2022_dir / "test-collection" / "B-Long-September" / "English"
dataset_2022_test_long_loader = LongEvalLoader(dataset_2022_test_long_dir, "test09.tsv", "b-long-september.txt")

dataset_2023_dir = longeval_dataset_dir / "2023"
dataset_2023_train_dir = dataset_2023_dir / "train_data" / "2023_01" / "English"
dataset_2023_train_main_loader = LongEvalLoader(dataset_2023_train_dir, "train.tsv")

dataset_2023_test_short_dir = dataset_2023_dir / "test_data" / "2023_06" / "English"
dataset_2023_test_short_loader = LongEvalLoader(dataset_2023_test_short_dir, "test.tsv")

dataset_2023_test_long_dir = dataset_2023_dir / "test_data" / "2023_08" / "English"
dataset_2023_test_long_loader = LongEvalLoader(dataset_2023_test_long_dir, "test.tsv")

loaders = {
    "2022": {
        "train": dataset_2022_train_main_loader,
        "test": {
            "short": dataset_2022_test_short_loader,
            "long": dataset_2022_test_long_loader
        }
    },
    "2023": {
        "train": dataset_2023_train_main_loader,
        "test": {
            "short": dataset_2023_test_short_loader,
            "long": dataset_2023_test_long_loader
        }
    }
}


def main():
    replace_with_xformers_attention()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--longeval", action="store_true", help="Whether to use long eval mode.")
    parser.add_argument('--year', type=int, default=2022, help='Year of the dataset to use. Default is 2022.')
    parser.add_argument('--split', type=str, default='train', help='Split of the dataset to use. Default is train.')
    parser.add_argument('--term', type=str, default='short', help='Term of the dataset to use. Default is short.')

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = RepLLaMA.load(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    year = other_args.year
    term = other_args.term
    dtype = other_args.split

    tmp_loader = loaders[str(year)][dtype]
    loader: LongEvalLoader = tmp_loader[term] if dtype == "test" else tmp_loader

    if data_args.encode_is_qry:
        if other_args.longeval:
            queries = loader.load_queries()
            # with open("/tmp/longeval_queries.jsonl", "w") as f:
            #     for qid, query in tqdm(queries.items()):
            #         f.write(json.dumps({
            #             "query_id": qid,
            #             "query": query
            #         }) + "\n")
            # data_args.custom_query_df = load_dataset("json", data_files="/tmp/longeval_queries.jsonl", split="train")
            data_args.custom_query_df = datasets.Dataset.from_list([
                {"query_id": qid, "query": query}
                for qid, query in queries.items()
            ])
            encode_dataset = CustomQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                                cache_dir=data_args.data_cache_dir or model_args.cache_dir)
        else:
            encode_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                            cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    else:
        if other_args.longeval:
            index_files = glob.glob(data_args.encoded_save_path.split("/")[0] + f"-rerank/corpus_{year}-{dtype}-{term}.*.pkl")
            p_reps_0, p_lookup_0 = pickle_load(index_files[0])
            shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards', total=len(index_files))
            already_encoded = []
            for p_reps, p_lookup in shards:
                already_encoded += p_lookup
            rankings_df = pd.read_csv(
                loader.data_dir / "bm25.ranking",
                sep=" ",
                header=None,
                names=["query_id", "Q0", "docid", "rank", "score", "tag"]
            )
            unique_docids = rankings_df["docid"].unique().tolist()
            print("already indexed:", len(already_encoded))
            print("unique docids:", len(unique_docids))
            print("diff: ", len(set(unique_docids) - set(already_encoded)))
            documents = loader.load_clean_documents(already_encoded)
            print("documents to index: ", len(documents))
            # with open("/tmp/longeval_documents.jsonl", "w") as f:
            #     for docid, document in tqdm(documents.items()):
            #         f.write(json.dumps({
            #             "docid": docid,
            #             "title": document["title"],
            #             "text": document["text"]
            #         }) + "\n")
            # data_args.custom_corpus_df = load_dataset(
            #     "json", data_files="/tmp/longeval_documents.jsonl", split="train",
            #     features=Features({
            #         "docid": Value("string"),
            #         "text": Value("string"),
            #         "title": Value("string"),
            #     }),
            # )
            data_args.custom_corpus_df = datasets.Dataset.from_list([
                {"docid": docid, "title": document["title"], "content": document["text"]}
                for docid, document in documents.items()
            ])

            encode_dataset = CustomCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                                 cache_dir=data_args.data_cache_dir or model_args.cache_dir)
        else:
            print("loading with HFCorpusDataset")
            encode_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                             cache_dir=data_args.data_cache_dir or model_args.cache_dir)

    encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                   tokenizer, max_len=text_max_length)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded)

    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
