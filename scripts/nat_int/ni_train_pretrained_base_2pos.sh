# NI training
# eval after since its a short experiment.

# name of experiment folder
EXPERIMENT_NAME=$1
LOAD_MODEL=$2
CHECKPOINT=$3
TRAIN_STEPS=$4
BUCKET_NAME="hamishi-tpu"

echo "Make sure train steps is set to > the checkpoint!"

# where model will be saved
MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_base.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="ni_train.gin" \
  --gin_file="full_restore.gin" \
  --gin.MIXTURE_OR_TASK_NAME=\"natural_instruction_positive_example_hyper\" \
  --gin.USE_CACHED_TASKS=True \
  --gin.trainer.Trainer.num_microbatches=16 \
  --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
  --gin.BATCH_SIZE=1024 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=$4 \
  --gin.partitioning.PjitPartitioner.num_partitions=2 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://${BUCKET_NAME}/$2/model/$3\"


echo "Training done. Now evaluating all checkpoints..."

EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval/"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_base.gin" \
    --gin_file="instruction_embed.gin" \
    --gin_file="ni_eval.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\"natural_instruction_positive_example_hyper\" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=256 \
    --gin.utils.DatasetConfig.split=\"test\" \
    --gin.partitioning.PjitPartitioner.num_partitions=2 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"all\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
