# name of experiment folder
EXPERIMENT_NAME=$1
BUCKET_NAME="hamishi-tpu"

# where model will be saved
MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xl.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="t0_train.gin" \
  --gin_file="partial_train_adafactor_no_roberta.gin" \
  --gin_file="hypter.gin" \
  --gin.hyper_network.HyperT5Config.adapter_size=4 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1120000 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl/checkpoint_1100000\"
