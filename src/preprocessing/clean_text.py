"""
    This file contains the code used to process and create the
    FineWeb dataset (https://huggingface.co/datasets/HuggingFaceFW/fineweb)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from cleantext import clean

from tqdm.rich import tqdm

from dataloader import LongEvalLoader
from paths import loaders

"""
    we first ran the following pipeline for each dump
"""

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--year', type=int, default=2022, help='year of test')
    arg_parser.add_argument('--term', type=str, default='short', help='short/long')
    arg_parser.add_argument('--dtype', type=str, default='train', help='train/test')
    arg_parser.add_argument('--output', type=str, default='clean_corpus.jsonl', help='output file')

    args = arg_parser.parse_args()

    year = args.year
    term = args.term
    dtype = args.dtype

    tmp_loader = loaders[str(year)][dtype]
    loader: LongEvalLoader = loaders[str(year)][dtype][term] if dtype == "test" else loaders[str(year)][dtype]

    output_path = loader.document_dir.parent / args.output

    cleanse = lambda text: clean(
        text,
        fix_unicode=True,  # fix various unicode errors
        to_ascii=True,  # transliterate to closest ASCII representation
        lower=False,  # lowercase text
        no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
        no_urls=True,  # replace all URLs with a special token
        no_emails=True,  # replace all email addresses with a special token
        no_phone_numbers=True,  # replace all phone numbers with a special token
        no_numbers=False,  # replace all numbers with a special token
        no_digits=False,  # replace all digits with a special token
        no_currency_symbols=False,  # replace all currency symbols with a special token
        no_punct=False,  # remove punctuations
        replace_with_punct="",  # instead of removing punctuations you may replace them
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"  # set to 'de' for German special handling
    )

    documents = loader.load_documents()
    with output_path.open("w") as fOut:
        for did, document in tqdm(documents.items(), desc="Cleaning documents"):
            save_obj = {
                "id": did,
                "title": cleanse(document["title"]),
                "text": cleanse(document["text"])
            }
            fOut.write(
                json.dumps(save_obj)
                + "\n"
            )

