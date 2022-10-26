# NI training
# eval after since its a short experiment.

# name of experiment folder
EXPERIMENT_NAME=$1

mkdir -p $EXPERIMENT_NAME/model
mkdir -p $EXPERIMENT_NAME/eval/inference_eval

# where model will be saved
MODEL_DIR="$EXPERIMENT_NAME/model" #"gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
JAX_DISABLE_JIT=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_base.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="ni_train.gin" \
  --gin_file="partial_train_adafactor.gin" \
  --gin.hyper_network.HyperT5Config.hyperencoder_model=\"google/t5-base-lm-adapt\" \
  --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions_hyper_alt\" \
  --gin.USE_CACHED_TASKS=True \
  --gin.trainer.Trainer.num_microbatches=1 \
  --gin.utils.create_learning_rate_scheduler.warmup_steps=100 \
  --gin.BATCH_SIZE=1024 \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1102000 \
  --gin.partitioning.PjitPartitioner.num_partitions=2 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000/\"

# echo "Training done. Now evaluating all checkpoints..."

gsutil -m cp -r $MODEL_DIR gs://hamishi-us-bucket/$EXPERIMENT_NAME/model/

# MODEL_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"
# EVAL_OUTPUT_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/eval/"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
#     --gin_search_paths="gins" \
#     --gin_file="hyper_base.gin" \
#     --gin_file="instruction_embed.gin" \
#     --gin_file="ni_eval.gin" \
#     --gin.hyper_network.HyperT5Config.hyperencoder_model=\"google/t5-base-lm-adapt\" \
#     --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions_hyper_alt\" \
#     --gin.USE_CACHED_TASKS=True \
#     --gin.utils.DatasetConfig.batch_size=512 \
#     --gin.utils.DatasetConfig.split=\"test\" \
#     --gin.partitioning.PjitPartitioner.num_partitions=2 \
#     --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
#     --gin.utils.RestoreCheckpointConfig.mode=\"all\" \
#     --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
