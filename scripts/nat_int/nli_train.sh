# NI training

# name of experiment folder
EXPERIMENT_NAME=$1
LOAD_MODEL=$2
CHECKPOINT=$3
TRAIN_STEPS=$4

# where model will be saved
MODEL_DIR="gs://hamishi-us-model/${EXPERIMENT_NAME}/model"

python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_small.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="nli_train.gin" \
  --gin_file="partial_train_adafactor.gin" \
  --gin.MIXTURE_OR_TASK_NAME=\"nli\" \
  --gin.USE_CACHED_TASKS=True \
  --gin.trainer.Trainer.num_microbatches=8 \
  --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
  --gin.BATCH_SIZE=1024 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=$4 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://hamishi-us-bucket/$2/model/$3\"