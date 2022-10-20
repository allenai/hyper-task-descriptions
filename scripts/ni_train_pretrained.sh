# NI training
# eval after since its a short experiment.

# name of experiment folder
EXPERIMENT_NAME=$1

# where model will be saved
MODEL_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_base.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="ni_train.gin" \
  --gin_file="partial_train_adam.gin" \
  --gin_file="full_restore.gin" \
  --gin.hyper_network.HyperT5Config.hyperencoder_model=\"google/t5-base-lm-adapt\" \
  --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions\" \
  --gin.USE_CACHED_TASKS=True \
  --gin.trainer.Trainer.num_microbatches=8 \
  --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
  --gin.BATCH_SIZE=1024 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1109000 \
  --gin.partitioning.PjitPartitioner.num_partitions=2 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://hamishi-us-bucket/pretrain-cross-att-linear-fixed/model/checkpoint_1108000\"

echo "Training done. Now evaluating all checkpoints..."

EVAL_OUTPUT_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/eval/"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_base.gin" \
    --gin_file="instruction_embed.gin" \
    --gin_file="ni_eval.gin" \
    --gin.hyper_network.HyperT5Config.hyperencoder_model=\"google/t5-base-lm-adapt\" \
    --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions\" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=512 \
    --gin.utils.DatasetConfig.split=\"test\" \
    --gin.partitioning.PjitPartitioner.num_partitions=2 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"all\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
