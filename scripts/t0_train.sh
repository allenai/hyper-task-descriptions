# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
EXPERIMENT_NAME=$1

MODEL_DIR="${EXPERIMENT_NAME}/model"

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="gs://hamishi-tpu-bucket/t0_data/data"
T5X_DIR="."  # directory where the T5X repo is cloned.
PROJECT_DIR="./hyper/gins"  # directory for extra bits
export PYTHONPATH=${PROJECT_DIR}

HF_DATASETS_OFFLINE=1 python3 t5x.train \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="enc_dec_t0_adapt.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
