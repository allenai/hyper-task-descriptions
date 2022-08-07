# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
EXPERIMENT_NAME=$1
CHECKPOINT_NAME=$2

# model checkpoint location
MODEL_DIR="gs://yizhongw-tpu-bucket/${EXPERIMENT_NAME}/model/${CHECKPOINT_NAME}"
# where to put eval results
EVAL_OUTPUT_DIR="gs://yizhongw-tpu-bucket/${EXPERIMENT_NAME}/eval/"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
# note you pass in a model file and the eval file.
python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_xl.gin" \
    --gin_file="ni_eval.gin" \
    --gin.USE_CACHED_TASKS=False \
    --gin.utils.DatasetConfig.batch_size=256 \
    --gin.utils.DatasetConfig.split=\"test\" \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"specific\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
