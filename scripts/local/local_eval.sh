# li'l script for testing eval stuff locally.
BUCKET_NAME="hamishi-tpu"

python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_xl.gin" \
    --gin_file="catwalk_eval.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\"eleuther::cola\" \
    --gin.EVAL_OUTPUT_DIR=\"model_eval_test\" \
    --gin.CHECKPOINT_PATH=\"gs://${BUCKET_NAME}/roberta_contrastive_dup_t0_3b/model/checkpoint_1123000\" \
    --gin.USE_CACHED_TASKS=False
