# commands

## lucene
### indexing
[Code Reference](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-embeddable-python-implementation)
```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input ./Json \
--index ./lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 8 \
--storePositions --storeDocvectors --storeRaw
```

### search
```bash
python -m pyserini.search.lucene \
  --index indexes/sample_collection_jsonl \
  --topics tests/resources/sample_queries.tsv \
  --output run.sample.txt \
  --bm25
```

## convert .txt to .json
```
for file in *.txt
do
mv "$file" "${file%.txt}.json"
done
```

## Actual commands

### 2022
```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input /workspace/longeval/datasets/2022/publish/English/Documents/clean/ \
--index /workspace/longeval/datasets/2022/publish/English/Documents/clean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/Documents/corpus/ \
--index /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/Documents/unclean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/Documents/corpus/ \
--index /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/Documents/unclean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```


### 2023
```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input /workspace/longeval/datasets/2023/train_data/2023_01/English/Documents/corpus/ \
--index /workspace/longeval/datasets/2023/train_data/2023_01/English/Documents/unclean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input /workspace/longeval/datasets/2023/test_data/2023_06/English/Documents/corpus/ \
--index /workspace/longeval/datasets/2023/test_data/2023_06/English/Documents/unclean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--collection JsonCollection \
--input /workspace/longeval/datasets/2023/test_data/2023_08/English/Documents/corpus/ \
--index /workspace/longeval/datasets/2023/test_data/2023_08/English/Documents/unclean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

# clean
### retrieve
```bash
sh ranking_bm25.sh /workspace/longeval/datasets/2022/publish/English/Documents/clean_lucene.index/ /workspace/longeval/datasets/2022/publish/English/Queries/train.tsv /workspace/longeval/datasets/2022/publish/English/bm25.ranking
sh ranking_bm25.sh /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/Documents/clean_lucene.index/ /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/Queries/test07.tsv /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/bm25.ranking
sh ranking_bm25.sh /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/Documents/clean_lucene.index/ /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/Queries/test09.tsv /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/bm25.ranking
```

```bash
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/train_data/2023_01/English/Documents/clean_lucene.index/ /workspace/longeval/datasets/2023/train_data/2023_01/English/Queries/train.tsv /workspace/longeval/datasets/2023/train_data/2023_01/English/bm25.ranking
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/test_data/2023_06/English/Documents/clean_lucene.index/ /workspace/longeval/datasets/2023/test_data/2023_06/English/Queries/test.tsv /workspace/longeval/datasets/2023/test_data/2023_06/English/bm25.ranking
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/test_data/2023_08/English/Documents/clean_lucene.index/ /workspace/longeval/datasets/2023/test_data/2023_08/English/Queries/test.tsv /workspace/longeval/datasets/2023/test_data/2023_08/English/bm25.ranking
```

### evaluate
```bash
sh evaluation.sh /workspace/longeval/datasets/2022/publish/English/bm25.ranking 2022 none train /workspace/longeval/datasets/2023/train_data/2023_01/English/bm25.evaluation
sh evaluation.sh /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/bm25.ranking 2022 short test /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/bm25.evaluation
sh evaluation.sh /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/bm25.ranking 2022 long test /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/bm25.evaluation

sh evaluation.sh /workspace/longeval/datasets/2023/train_data/2023_01/English/bm25.ranking 2023 none train /workspace/longeval/datasets/2023/train_data/2023_01/English/bm25.evaluation
sh evaluation.sh /workspace/longeval/datasets/2023/test_data/2023_06/English/bm25.ranking 2023 short test /workspace/longeval/datasets/2023/test_data/2023_06/English/bm25.evaluation
sh evaluation.sh /workspace/longeval/datasets/2023/test_data/2023_08/English/bm25.ranking 2023 long test /workspace/longeval/datasets/2023/test_data/2023_08/English/bm25.evaluation
```

# unclean
### retrieve
```bash
sh ranking_bm25.sh /workspace/longeval/datasets/2022/publish/English/Documents/unclean_lucene.index/ /workspace/longeval/datasets/2022/publish/English/Queries/train.tsv /workspace/longeval/datasets/2022/publish/English/unclean_bm25.ranking
sh ranking_bm25.sh /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/Documents/unclean_lucene.index/ /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/Queries/test07.tsv /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/unclean_bm25.ranking
sh ranking_bm25.sh /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/Documents/unclean_lucene.index/ /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/Queries/test09.tsv /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/unclean_bm25.ranking
```

```bash
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/train_data/2023_01/English/Documents/unclean_lucene.index/ /workspace/longeval/datasets/2023/train_data/2023_01/English/Queries/train.tsv /workspace/longeval/datasets/2023/train_data/2023_01/English/unclean_bm25.ranking
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/test_data/2023_06/English/Documents/unclean_lucene.index/ /workspace/longeval/datasets/2023/test_data/2023_06/English/Queries/test.tsv /workspace/longeval/datasets/2023/test_data/2023_06/English/unclean_bm25.ranking
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/test_data/2023_08/English/Documents/unclean_lucene.index/ /workspace/longeval/datasets/2023/test_data/2023_08/English/Queries/test.tsv /workspace/longeval/datasets/2023/test_data/2023_08/English/unclean_bm25.ranking
```

### evaluate
```bash
sh evaluation.sh /workspace/longeval/datasets/2022/publish/English/unclean_bm25.ranking 2022 none train /workspace/longeval/datasets/2022/publish/English/unclean_bm25.evaluation false
sh evaluation.sh /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/unclean_bm25.ranking 2022 short test /workspace/longeval/datasets/2022/test-collection/A-Short-July/English/unclean_bm25.evaluation false
sh evaluation.sh /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/unclean_bm25.ranking 2022 long test /workspace/longeval/datasets/2022/test-collection/B-Long-September/English/unclean_bm25.evaluation false

sh evaluation.sh /workspace/longeval/datasets/2023/train_data/2023_01/English/unclean_bm25.ranking 2023 none train /workspace/longeval/datasets/2023/train_data/2023_01/English/unclean_bm25.evaluation false
sh evaluation.sh /workspace/longeval/datasets/2023/test_data/2023_06/English/unclean_bm25.ranking 2023 short test /workspace/longeval/datasets/2023/test_data/2023_06/English/unclean_bm25.evaluation false
sh evaluation.sh /workspace/longeval/datasets/2023/test_data/2023_08/English/unclean_bm25.ranking 2023 long test /workspace/longeval/datasets/2023/test_data/2023_08/English/unclean_bm25.evaluation false
```



## Paths
Server: cluster/n03
### clean_corpus (text)
fields: id, title (empty), text, *some_other_metas
2022-train
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2022/publish/English/Documents/clean_corpus.jsonl`
2022-test-short
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2022/test-collection/A-Short-July/English/Documents/clean_corpus.jsonl`
2022-test-long
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2022/test-collection/B-Long-September/English/Documents/clean_corpus.jsonl`

2023-train
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2023/train_data/2023_01/English/Documents/clean_corpus.jsonl`
2023-test-short
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2023/test_data/2023_06/English/Documents/clean_corpus.jsonl`
2023-test-long
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2023/test_data/2023_08/English/Documents/clean_corpus.jsonl`

### bm25 ranking
2022-train
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2022/publish/English/bm25.ranking`
2023-train
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2023/train_data/2023_01/English/bm25.ranking`

### bm25 evaluation
2022-train
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2022/publish/English/bm25.evaluation`
2023-train
`/data_n03_sda1/jongyoon/docker_volume/longeval/datasets/2023/train_data/2023_01/English/bm25.evaluation`

### bm25 evaluation results
#### 2022-train-evaluation
~~~clean
       NDCG    MAP     Recall  P
@1      0.1114  0.0381  0.0381  0.1480
@3      0.1220  0.0703  0.0973  0.1241
@5      0.1341  0.0868  0.1439  0.1094
@10     0.1676  0.1108  0.2268  0.0885
@20     0.1968  0.1234  0.3056  0.0607
@50     0.2260  0.1315  0.4063  0.0325
@100    0.2420  0.1341  0.4760  0.0191
~~~
~~~unclean
	NDCG	MAP	Recall	P
@1	0.1140	0.0395	0.0395	0.1520
@3	0.1237	0.0715	0.0979	0.1262
@5	0.1354	0.0882	0.1442	0.1106
@10	0.1688	0.1125	0.2272	0.0888
@20	0.1975	0.1247	0.3054	0.0608
@50	0.2267	0.1328	0.4069	0.0325
@100	0.2423	0.1353	0.4748	0.0191
~~~
#### 2022-test-short
~~~clean
	NDCG	MAP	Recall	P
@1	0.1209	0.0401	0.0401	0.1517
@3	0.1307	0.0785	0.1076	0.1364
@5	0.1406	0.0952	0.1506	0.1140
@10	0.1766	0.1195	0.2404	0.0919
@20	0.2083	0.1338	0.3258	0.0632
@50	0.2334	0.1411	0.4160	0.0327
@100	0.2502	0.1436	0.4879	0.0191
~~~
~~~unclean
	NDCG	MAP	Recall	P
@1	0.1184	0.0402	0.0402	0.1503
@3	0.1292	0.0781	0.1066	0.1355
@5	0.1396	0.0950	0.1510	0.1128
@10	0.1747	0.1189	0.2376	0.0911
@20	0.2054	0.1325	0.3212	0.0620
@50	0.2307	0.1398	0.4108	0.0321
@100	0.2472	0.1423	0.4809	0.0189
~~~
#### 2022-test-long
~~~clean
	NDCG	MAP	Recall	P
@1	0.1162	0.0339	0.0339	0.1487
@3	0.1287	0.0698	0.0968	0.1354
@5	0.1397	0.0897	0.1452	0.1225
@10	0.1738	0.1151	0.2292	0.0983
@20	0.2072	0.1305	0.3225	0.0681
@50	0.2368	0.1390	0.4228	0.0363
@100	0.2546	0.1419	0.4969	0.0215
~~~
~~~unclean
	NDCG	MAP	Recall	P
@1	0.1162	0.0346	0.0346	0.1509
@3	0.1278	0.0698	0.0968	0.1354
@5	0.1381	0.0887	0.1431	0.1205
@10	0.1737	0.1149	0.2300	0.0986
@20	0.2059	0.1296	0.3205	0.0675
@50	0.2359	0.1381	0.4219	0.0362
@100	0.2527	0.1408	0.4931	0.0212
~~~
#### 2023-train-evaluation
~~~clean
       NDCG    MAP     Recall  P
@1      0.1449  0.0322  0.0322  0.2278
@3      0.1551  0.0670  0.0867  0.2038
@5      0.1567  0.0883  0.1281  0.1806
@10     0.1848  0.1191  0.2146  0.1507
@20     0.2265  0.1411  0.3102  0.1104
@50     0.2765  0.1602  0.4532  0.0657
@100    0.3034  0.1666  0.5463  0.0398
~~~
~~~unclean
	NDCG	MAP	Recall	P
@1	0.1382	0.0306	0.0306	0.2161
@3	0.1526	0.0656	0.0859	0.2021
@5	0.1542	0.0865	0.1267	0.1786
@10	0.1794	0.1155	0.2083	0.1466
@20	0.2216	0.1374	0.3040	0.1084
@50	0.2706	0.1558	0.4437	0.0644
@100	0.2969	0.1622	0.5353	0.0391
~~~

# 05/06
## SPLADE encoding (tmux 4)
```shard
CUDA_VISIBLE_DEVICES=0 python -m pyserini.encode \
  input   --corpus ./datasets/2022/publish/English/Documents/clean_corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ./datasets/2022/publish/English/Documents/corpus/splade_v2_max/embeddings.jsonl-0 \
  encoder --encoder naver/splade_v2_max \
          --encoder-class auto \
          --fields text \
          --batch 32 \
          --fp16
CUDA_VISIBLE_DEVICES=2 python -m pyserini.encode \
  input   --corpus ./datasets/2022/publish/English/Documents/clean_corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 1 \
          --shard-num 3 \
  output  --embeddings ./datasets/2022/publish/English/Documents/corpus/splade_v2_max/embeddings.jsonl-1 \
  encoder --encoder naver/splade_v2_max \
          --fields text \
          --batch 32 \
          --fp16
CUDA_VISIBLE_DEVICES=3 python -m pyserini.encode \
  input   --corpus ./datasets/2022/publish/English/Documents/clean_corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 2 \
          --shard-num 3 \
  output  --embeddings ./datasets/2022/publish/English/Documents/corpus/splade_v2_max/embeddings.jsonl-2 \
  encoder --encoder naver/splade_v2_max \
          --fields text \
          --batch 32 \
          --fp16
```
```
python -m pyserini.encode \
  input   --corpus ./datasets/2022/publish/English/Documents/clean_corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ./datasets/2022/publish/English/Documents/corpus/splade_v2_max/embeddings.jsonl \
  encoder --encoder naver/splade_v2_max \
          --fields text \
          --batch 32 \
          --fp16
          
CUDA_VISIBLE_DEVICES=2 python -m pyserini.encode \
  input   --corpus ./datasets/2022/test-collection/A-Short-July/English/Documents/clean_corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ./datasets/2022/test-collection/A-Short-July/English/Documents/corpus/splade_v2_max/embeddings.jsonl \
          --to-faiss \
  encoder --encoder naver/splade_v2_max \
          --fields text \
          --batch 32 \
          --fp16
          
CUDA_VISIBLE_DEVICES=3 python -m pyserini.encode \
  input   --corpus ./datasets/2022/test-collection/B-Long-September/English/Documents/clean_corpus.jsonl \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ./datasets/2022/test-collection/B-Long-September/English/Documents/corpus/splade_v2_max/embeddings.jsonl \
          --to-faiss \
  encoder --encoder naver/splade_v2_max \
          --fields text \
          --batch 32 \
          --fp16
```

## SPLADE encode query
```
mkdir -p ./datasets/2022/publish/English/Queries/splade/ ;
python -m pyserini.encode.query \
  --topics ./datasets/2022/publish/English/Queries/train.tsv \
  --encoder naver/splade_v2_max \
  --output ./datasets/2022/publish/English/Queries/splade/encoded_queries.jsonl \
  --weight-range 1 --quant-range 1 --max-length
```
```
python -m pyserini.search.faiss \
  --index ./datasets/2022/publish/English/Documents/corpus/splade_v2_max/embeddings.jsonl \
  --topics ./datasets/2022/publish/English/Queries/train.tsv \
  --encoded-queries ./datasets/2022/publish/English/Queries/splade/encoded_queries.jsonl \
  --output ./datasets/2022/publish/English/runs/run.longeval-passage-distill-splade-max.tsv \
  --output-format msmarco \
  --batch-size 36 --threads 12
```

## SPLADE indexing (tmux 4)
`CUDA_VISIBLE_DEVICES=0 python -m pyserini.index.lucene   --collection JsonCollection   --input ./datasets/2022/publish/English/Documents/corpus   --index ./datasets/2022/publish/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`
`CUDA_VISIBLE_DEVICES=2 python -m pyserini.index.lucene   --collection JsonCollection   --input ./datasets/2022/test-collection/A-Short-July/English/Documents/corpus   --index ./datasets/2022/test-collection/A-Short-July/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`
`CUDA_VISIBLE_DEVICES=3 python -m pyserini.index.lucene   --collection JsonCollection   --input ./datasets/2022/test-collection/B-Long-September/English/Documents/corpus   --index ./datasets/2022/test-collection/B-Long-September/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`

`CUDA_VISIBLE_DEVICES=2 python -m pyserini.index.lucene   --collection JsonCollection   --input ./datasets/2023/train_data/2023_01/English/Documents/corpus   --index ./datasets/2023/train_data/2023_01/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`
`CUDA_VISIBLE_DEVICES=0 python -m pyserini.index.lucene   --collection JsonCollection   --input ./datasets/2023/test_data/2023_06/English/Documents/corpus   --index ./datasets/2023/test_data/2023_06/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`
`CUDA_VISIBLE_DEVICES=3 python -m pyserini.index.lucene   --collection JsonCollection   --input ./datasets/2023/test_data/2023_08/English/Documents/corpus   --index ./datasets/2023/test_data/2023_08/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`

`CUDA_VISIBLE_DEVICES=0 python -m pyserini.index.lucene   --collection JsonVectorCollection   --input ./datasets/2023/test_data/2023_06/English/Documents/splade_corpus   --index ./datasets/2023/test_data/2023_06/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`
`CUDA_VISIBLE_DEVICES=3 python -m pyserini.index.lucene   --collection JsonVectorCollection   --input ./datasets/2023/test_data/2023_08/English/Documents/splade_corpus   --index ./datasets/2023/test_data/2023_08/English/indexes/lucene-index.longeval-passage-distill-splade-max   --generator DefaultLuceneDocumentGenerator   --threads 36 --impact --pretokenized`

## SPLADE retrieval (tmux 4)
```
# 2022
python -m pyserini.search.lucene \
  --index ./datasets/2022/publish/English/Documents/corpus/splade_v2_max/embeddings.jsonl/ \
  --topics ./datasets/2022/publish/English/Queries/train.tsv \
  --encoder naver/splade_v2_max \
  --output ./datasets/2022/publish/English/runs/run.longeval-passage-distill-splade-max.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
  
python -m pyserini.eval.convert_msmarco_run_to_trec_run \
    --input ./datasets/2022/publish/English/runs/run.longeval-passage-distill-splade-max.tsv \
    --output ./datasets/2022/publish/English/runs/run.longeval-passage-distill-splade-max.trec; \
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 -m recall.1000 \
    ./datasets/2022/publish/English/Qrels/train.txt \
    ./datasets/2022/publish/English/runs/run.longeval-passage-distill-splade-max.trec
  
python -m pyserini.search.lucene \
  --index ./datasets/2022/test-collection/A-Short-July/English/indexes/lucene-index.longeval-passage-distill-splade-max \
  --topics ./datasets/2022/test-collection/A-Short-July/English/Queries/test07.tsv \
  --encoder naver/splade_v2_max \
  --output ./datasets/2022/test-collection/A-Short-July/English/runs/run.longeval-passage-distill-splade-max.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact

python -m pyserini.search.lucene \
  --index ./datasets/2022/test-collection/B-Long-September/English/indexes/lucene-index.longeval-passage-distill-splade-max \
  --topics ./datasets/2022/test-collection/B-Long-September/English/Queries/test09.tsv \
  --encoder naver/splade_v2_max \
  --output ./datasets/2022/test-collection/B-Long-September/English/runs/run.longeval-passage-distill-splade-max.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
  
# 2023
python -m pyserini.search.lucene \
  --index ./datasets/2023/train_data/2023_01/English/indexes/lucene-index.longeval-passage-distill-splade-max \
  --topics ./datasets/2023/train_data/2023_01/English/Queries/train.tsv \
  --encoder naver/splade_v2_max \
  --output ./datasets/2023/train_data/2023_01/English/runs/run.longeval-passage-distill-splade-max.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
  
python -m pyserini.search.lucene \
  --index ./datasets/2023/test_data/2023_06/English/indexes/lucene-index.longeval-passage-distill-splade-max \
  --topics ./datasets/2023/test_data/2023_06/English/Queries/test.tsv \
  --encoder naver/splade_v2_max \
  --output ./datasets/2023/test_data/2023_06/English/runs/run.longeval-passage-distill-splade-max.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
  
python -m pyserini.search.lucene \
  --index ./datasets/2023/test_data/2023_08/English/indexes/lucene-index.longeval-passage-distill-splade-max \
  --topics ./datasets/2023/test_data/2023_08/English/Queries/test.tsv \
  --encoder naver/splade_v2_max \
  --output ./datasets/2023/test_data/2023_08/English/runs/run.longeval-passage-distill-splade-max.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
```

## summary, time gen ()
* time: tmux 4
* summary: tmux 5

# 05/07
## French
### 2022
```bash
python -m pyserini.index.lucene \
--language fr \
--collection JsonCollection \
--input /workspace/longeval/datasets/2022/publish/French/Documents/clean/ \
--index /workspace/longeval/datasets/2022/publish/French/Documents/clean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--language fr \
--collection JsonCollection \
--input /workspace/longeval/datasets/2022/test-collection/A-Short-July/French/Documents/clean/ \
--index /workspace/longeval/datasets/2022/test-collection/A-Short-July/French/Documents/clean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--language fr \
--collection JsonCollection \
--input /workspace/longeval/datasets/2022/test-collection/B-Long-September/French/Documents/clean/ \
--index /workspace/longeval/datasets/2022/test-collection/B-Long-September/French/Documents/clean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```


### 2023
```bash
python -m pyserini.index.lucene \
--language fr \
--collection JsonCollection \
--input /workspace/longeval/datasets/2023/train_data/2023_01/French/Documents/clean/ \
--index /workspace/longeval/datasets/2023/train_data/2023_01/French/Documents/clean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--language fr \
--collection JsonCollection \
--input /workspace/longeval/datasets/2023/test_data/2023_06/French/Documents/clean/ \
--index /workspace/longeval/datasets/2023/test_data/2023_06/French/Documents/clean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

```bash
python -m pyserini.index.lucene \
--language fr \
--collection JsonCollection \
--input /workspace/longeval/datasets/2023/test_data/2023_08/French/Documents/clean/ \
--index /workspace/longeval/datasets/2023/test_data/2023_08/French/Documents/clean_lucene.index \
--generator DefaultLuceneDocumentGenerator \
--threads 36 \
--storePositions --storeDocvectors --storeRaw
```

# clean
### retrieve
```bash
sh ranking_bm25.sh /workspace/longeval/datasets/2022/publish/French/Documents/clean_lucene.index/ /workspace/longeval/datasets/2022/publish/French/Queries/train.tsv /workspace/longeval/datasets/2022/publish/French/bm25.ranking
sh ranking_bm25.sh /workspace/longeval/datasets/2022/test-collection/A-Short-July/French/Documents/clean_lucene.index/ /workspace/longeval/datasets/2022/test-collection/A-Short-July/French/Queries/test07.tsv /workspace/longeval/datasets/2022/test-collection/A-Short-July/French/bm25.ranking
sh ranking_bm25.sh /workspace/longeval/datasets/2022/test-collection/B-Long-September/French/Documents/clean_lucene.index/ /workspace/longeval/datasets/2022/test-collection/B-Long-September/French/Queries/test09.tsv /workspace/longeval/datasets/2022/test-collection/B-Long-September/French/bm25.ranking
```

```bash
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/train_data/2023_01/French/Documents/clean_lucene.index/ /workspace/longeval/datasets/2023/train_data/2023_01/French/Queries/train.tsv /workspace/longeval/datasets/2023/train_data/2023_01/French/bm25.ranking
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/test_data/2023_06/French/Documents/clean_lucene.index/ /workspace/longeval/datasets/2023/test_data/2023_06/French/Queries/test.tsv /workspace/longeval/datasets/2023/test_data/2023_06/French/bm25.ranking
sh ranking_bm25.sh  /workspace/longeval/datasets/2023/test_data/2023_08/French/Documents/clean_lucene.index/ /workspace/longeval/datasets/2023/test_data/2023_08/French/Queries/test.tsv /workspace/longeval/datasets/2023/test_data/2023_08/French/bm25.ranking
```

### evaluate
```bash
sh evaluation_french.sh /workspace/longeval/datasets/2022/publish/French/bm25.ranking 2022 none train /workspace/longeval/datasets/2022/publish/French/bm25.evaluation true
sh evaluation_french.sh /workspace/longeval/datasets/2022/test-collection/A-Short-July/French/bm25.ranking 2022 short test /workspace/longeval/datasets/2022/test-collection/A-Short-July/French/bm25.evaluation true
sh evaluation_french.sh /workspace/longeval/datasets/2022/test-collection/B-Long-September/French/bm25.ranking 2022 long test /workspace/longeval/datasets/2022/test-collection/B-Long-September/French/bm25.evaluation true

sh evaluation_french.sh /workspace/longeval/datasets/2023/train_data/2023_01/French/bm25.ranking 2023 none train /workspace/longeval/datasets/2023/train_data/2023_01/French/bm25.evaluation true
sh evaluation_french.sh /workspace/longeval/datasets/2023/test_data/2023_06/French/bm25.ranking 2023 short test /workspace/longeval/datasets/2023/test_data/2023_06/French/bm25.evaluation true
sh evaluation_french.sh /workspace/longeval/datasets/2023/test_data/2023_08/French/bm25.ranking 2023 long test /workspace/longeval/datasets/2023/test_data/2023_08/French/bm25.evaluation true
```
