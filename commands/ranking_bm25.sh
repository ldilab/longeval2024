INDEX_PATH=$1
QUERIES_PATH=$2
OUTPUT_PATH=$3

python -m pyserini.search.lucene \
--index $INDEX_PATH \
--topics $QUERIES_PATH \
--output $OUTPUT_PATH \
--bm25