# name of experiment folder
EXPERIMENT_NAME=$1

# where model will be saved
MODEL_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xxl.gin" \
  --gin_file="t0_train.gin" \
  --gin.hyper_network.HyperT5Config.use_adapter=False \
  --gin.hyper_network.HyperT5Config.use_prefix=False \
  --gin.hyper_network.HyperT5Config.use_instructions=False \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1212200 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.trainer.Trainer.num_microbatches=128 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xxl/checkpoint_1100000\" \
  --tfds_data_dir="gs://hamishi-us-bucket/t0_data/data"
