# name of experiment folder
EXPERIMENT_NAME=$1
LOAD_MODEL=$2
CHECKPOINT=$3
TRAIN_STEPS=$4

# where model will be saved
MODEL_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xl.gin" \
  --gin_file="t0_train.gin" \
  --gin_file="partial_train_adafactor_dual.gin" \
  --gin_file="full_restore.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=$4 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://hamishi-us-bucket/$2/model/$3\"
