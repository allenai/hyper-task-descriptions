# NI training

# name of experiment folder
checkpoint=$1

# where model will be saved
MODEL_DIR="test/model"

# JAX_DISABLE_JIT=1 python3 -m t5x.train \
#   --gin_search_paths=gins \
#   --gin_file="hyper_xl.gin" \
#   --gin_file="instruction_embed.gin" \
#   --gin_file="ni_train.gin" \
#   --gin_file="partial_train_adafactor.gin" \
#   --gin_file="full_restore.gin" \
#   --gin.USE_CACHED_TASKS=True \
#   --gin.trainer.Trainer.num_microbatches=1 \
#   --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
#   --gin.BATCH_SIZE=2 \
#   --gin.MODEL_DIR=\"${MODEL_DIR}\" \
#   --gin.TRAIN_STEPS=1170000 \
#   --gin.partitioning.PjitPartitioner.num_partitions=8 \
#   --gin.partitioning.PjitPartitioner.use_cpu_pjit=True \
#   --gin.train.use_gda=False \
#   --gin.INITIAL_CHECKPOINT_PATH=\"checkpoint_1111000\"



JAX_DISABLE_JIT=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xl.gin" \
  --gin_file="ni_train.gin" \
  --gin_file="partial_train_adafactor.gin" \
  --gin_file="full_restore.gin" \
  --gin.USE_CACHED_TASKS=True \
  --gin.hyper_network.HyperT5Config.use_adapter=False \
  --gin.hyper_network.HyperT5Config.use_prefix=False \
  --gin.hyper_network.HyperT5Config.use_instructions=False \
  --gin.trainer.Trainer.num_microbatches=1 \
  --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
  --gin.BATCH_SIZE=2 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1170000 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.partitioning.PjitPartitioner.use_cpu_pjit=True \
  --gin.train.use_gda=False \
  --gin.INITIAL_CHECKPOINT_PATH=\"checkpoint_1101000\"
  
#\"${checkpoint}\"
#   --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions\" \

# echo "Training done. Now evaluating all checkpoints..."


# EVAL_OUTPUT_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/eval/"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
#     --gin_search_paths="gins" \
#     --gin_file="hyper_xl.gin" \
#     --gin_file="instruction_embed.gin" \
#     --gin_file="ni_eval.gin" \
#     --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions\" \
#     --gin.USE_CACHED_TASKS=True \
#     --gin.utils.DatasetConfig.batch_size=512 \
#     --gin.utils.DatasetConfig.split=\"test\" \
#     --gin.partitioning.PjitPartitioner.num_partitions=8 \
#     --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
#     --gin.utils.RestoreCheckpointConfig.mode=\"all\" \
#     --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
