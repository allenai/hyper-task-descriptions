EXPERIMENT_NAME=$1

# where model will be saved
MODEL_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"

python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xxl.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="pretrain.gin" \
  --gin_file="partial_train_adafactor.gin" \
  --gin.hyper_network.HyperT5Config.hyperencoder_model=\"google/t5-base-lm-adapt\" \
  --gin.USE_CACHED_TASKS=True \
  --gin.trainer.Trainer.num_microbatches=32 \
  --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
  --gin.BATCH_SIZE=1024 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1120000 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xxl/checkpoint_1100000/\"