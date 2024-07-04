import json
from collections import defaultdict
from pathlib import Path

from tqdm.rich import tqdm


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

    def load_clean_documents(self):
        n_lines_clean_document = sum(1 for _ in self.clean_document_path.open())
        documents = {}
        with self.clean_document_path.open("r") as f:
            for line in tqdm(f, total=n_lines_clean_document):
                document = json.loads(line)
                doc_id = document["id"]
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
