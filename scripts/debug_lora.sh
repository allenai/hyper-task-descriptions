# name of experiment folder
EXPERIMENT_NAME=$1

# where model will be saved
# MODEL_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"
MODEL_DIR="${EXPERIMENT_NAME}"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="lora/plain/lora_small.gin" \
  --gin_file="t0_train_local.gin" \
  --gin_file="partial_train_adam.gin" \
  --gin_file="lora/lora.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1101000 \
  --gin.partitioning.PjitPartitioner.num_partitions=1 \
  --gin.hyper_network.HyperT5Config.lora_ranks="(4,None,4,None)" \
  --gin.utils.create_learning_rate_scheduler.base_learning_rate=1e-4 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_small/checkpoint_1100000\" \
  --tfds_data_dir="gs://hamishi-us-bucket/t0_data/data"
