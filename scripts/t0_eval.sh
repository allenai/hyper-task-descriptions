# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
EXPERIMENT_NAME=$1

# model checkpoint location
MODEL_DIR="gs://hamishi-tpu-bucket/${EXPERIMENT_NAME}/model"
# where to put eval results
EVAL_OUTPUT_DIR="gs://hamishi-tpu-bucket/${EXPERIMENT_NAME}/eval"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
# note you pass in a model file and the eval file.
HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_xl.gin" \
    --gin_file="t0_eval.gin" \
    --gin.utils.DatasetConfig.batch_size=128 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
