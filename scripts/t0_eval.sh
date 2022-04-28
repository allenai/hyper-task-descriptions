# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
BUCKET_NAME=$1
EXPERIMENT_NAME=$2

MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model"
EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval_output"
T5X_DIR="."  # directory where the T5X repo is cloned.
PROJECT_DIR="./hyper/gins"  # directory for extra bits
export PYTHONPATH=${PROJECT_DIR}


HF_DATASETS_OFFLINE=1 python3 ${T5X_DIR}/t5x/eval.py \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="enc_dec_xxl.gin" \
    --gin_file="eval_t0.gin" \
    --gin.utils.DatasetConfig.batch_size=128 \
    --gin.CHECKPOINT_PATH="'$MODEL_DIR'" \
    --gin.EVAL_OUTPUT_DIR="'$EVAL_OUTPUT_DIR'"
