set -e;


cuda=$1
year=$2
split=$3
term=$4
dataset=${year}-${split}-${term}

mkdir -p longeval_embedding_${dataset}

# encode documents
for s in 0 1 2 3;
do
    if [ -f ./longeval_embedding_${dataset}/corpus_${dataset}.${s}.pkl ]; then
        echo "document pkl exists"
    else
        CUDA_VISIBLE_DEVICES=${cuda} python encode.py \
          --longeval \
          --year ${year} \
          --split ${split} \
          --term ${term} \
          --output_dir=temp \
          --model_name_or_path castorini/repllama-v1-7b-lora-passage \
          --tokenizer_name meta-llama/Llama-2-7b-hf \
          --fp16 \
          --per_device_eval_batch_size 16 \
          --p_max_len 512 \
          --dataset_name Tevatron/longeval-corpus:${dataset} \
          --encoded_save_path longeval_embedding_${dataset}/corpus_${dataset}.${s}.pkl \
          --encode_num_shard 4 \
          --encode_shard_index ${s}

    fi
done



## convert to TREC format
#python -m tevatron.utils.format.convert_result_to_trec \
#    --input longeval_embedding_${dataset}/rank.${dataset}.txt \
#    --output longeval_embedding_${dataset}/rank.${dataset}.trec \
#    --remove_query
#
## Evaluate
#python -m pyserini.eval.trec_eval -c \
#    -mrecall.100 \
#    -mndcg_cut.10 \
#    longeval-v1.0.0-${dataset}-test \
#    longeval_embedding_${dataset}/rank.${dataset}.trec
