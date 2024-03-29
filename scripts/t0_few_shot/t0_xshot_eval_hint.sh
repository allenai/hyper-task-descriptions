# name of experiment folder
EXPERIMENT_NAME=$1
SHOT=$2  # must be 1, 2, 4, 5
BUCKET_NAME="hamishi-tpu"

# where model will be saved
MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model"
EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_xl.gin" \
    --gin_file="t0_eval.gin" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=128 \
    --gin.MIXTURE_OR_TASK_NAME=\"t0_eval_score_eval_${SHOT}_shot\" \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"all\"
