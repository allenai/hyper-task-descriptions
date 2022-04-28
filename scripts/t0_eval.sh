# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
BUCKET_NAME=$1
EXPERIMENT_NAME=$2

MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model"
EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval_output"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
# note you pass in a model file and the eval file.
HF_DATASETS_OFFLINE=1 python -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_search_paths="t5x/examples/t5/t5_1_1" \
    --gin_file="xl.gin" \
    --gin_file="t0_eval.gin" \
    --gin.utils.DatasetConfig.batch_size=128 \
    --gin.CHECKPOINT_PATH="'$MODEL_DIR'" \
    --gin.EVAL_OUTPUT_DIR="'$EVAL_OUTPUT_DIR'"
