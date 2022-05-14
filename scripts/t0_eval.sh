# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
EXPERIMENT_NAME=$1

# model checkpoint location
MODEL_DIR="gs://bigscience/experiment_d/finetune-t5-xl-lm-d4-091621/model.ckpt-1100000"
# where to put eval results
EVAL_OUTPUT_DIR="gs://hamishi-tpu-bucket/t0_3b_eval_output"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
# note you pass in a model file and the eval file.
HF_DATASETS_OFFLINE=0 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="t5x/examples/t5/t5_1_1/xl.gin" \
    --gin_file="t0_eval.gin" \
    --gin.utils.DatasetConfig.batch_size=32 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
