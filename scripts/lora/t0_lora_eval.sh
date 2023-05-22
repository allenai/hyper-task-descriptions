# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
EXPERIMENT_NAME=$1
CHECKPOINT_NAME=$2
BUCKET_NAME="hamishi-tpu"

# model checkpoint location
MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model/${CHECKPOINT_NAME}"
# where to put eval results
EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval"

#MODEL_DIR="plain-lora-small-4n4n/model/checkpoint_1107000"
#EVAL_OUTPUT_DIR="plain-lora-small-4n4n-eval"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
# note you pass in a model file and the eval file.
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="lora/plain/lora_xl.gin" \
    --gin_file="t0_eval.gin" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=64 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
