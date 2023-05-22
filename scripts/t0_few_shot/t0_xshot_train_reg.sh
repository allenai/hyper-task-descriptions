# name of experiment folder
EXPERIMENT_NAME=$1
SHOT=$2  # must be 1, 2, 4, 5
BUCKET_NAME="hamishi-tpu"

# where model will be saved
MODEL_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/model"
EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xl.gin" \
  --gin_file="t0_train.gin" \
  --gin.hyper_network.HyperT5Config.use_adapter=False \
  --gin.hyper_network.HyperT5Config.use_prefix=False \
  --gin.hyper_network.HyperT5Config.use_instructions=False \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.MIXTURE_OR_TASK_NAME=\"t0_train_${SHOT}_shot\" \
  --gin.TRAIN_STEPS=1110000 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl/checkpoint_1100000\"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_xl.gin" \
    --gin_file="t0_eval.gin" \
    --gin.USE_CACHED_TASKS=True \
    --gin.hyper_network.HyperT5Config.use_adapter=False \
    --gin.hyper_network.HyperT5Config.use_prefix=False \
    --gin.hyper_network.HyperT5Config.use_instructions=False \
    --gin.utils.DatasetConfig.batch_size=128 \
    --gin.MIXTURE_OR_TASK_NAME=\"t0_eval_score_eval_${SHOT}_shot\" \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"all\"
