# Checkpoint to eval on
MODEL_DIR=$1
SAVE_DIR=$2

# where to put eval results
EVAL_OUTPUT_DIR="gs://hamishi-us-bucket/${SAVE_DIR}"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
# note you pass in a model file and the eval file.
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="t5x/examples/t5/t5_1_1/xl.gin" \
    --gin_file="t0_eval.gin" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=128 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
