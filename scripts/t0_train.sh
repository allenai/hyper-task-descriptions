# name of experiment folder
EXPERIMENT_NAME=$1

# where model will be saved
MODEL_DIR="gs://hamishi-tpu-bucket/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
HF_DATASETS_OFFLINE=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_small.gin" \
  --gin_file="partial_train.gin" \
  --gin_file="t0_train.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1105000 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_small/checkpoint_1100000\" \
  --tfds_data_dir="gs://hamishi-tpu-bucket/t0_data/data"
