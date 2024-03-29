# NI training

# name of experiment folder
EXPERIMENT_NAME=$1
BUCKET_NAME="hamishi-tpu"

# where model will be saved
MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model"

python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xl.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="ni_train_mixed.gin" \
  --gin_file="partial_train_adafactor_dual.gin" \
  --gin.MIXTURE_OR_TASK_NAME=\"c4_ni\" \
  --gin.USE_CACHED_TASKS=True \
  --gin.trainer.Trainer.num_microbatches=16 \
  --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
  --gin.BATCH_SIZE=1024 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1101000 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl/checkpoint_1100000/\"

echo "Training done. Now evaluating all checkpoints..."


EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval/"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_xl.gin" \
    --gin_file="instruction_embed.gin" \
    --gin_file="ni_eval.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions\" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=512 \
    --gin.utils.DatasetConfig.split=\"test\" \
    --gin.partitioning.PjitPartitioner.num_partitions=8 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"all\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
