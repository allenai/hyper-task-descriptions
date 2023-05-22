# name of experiment folder
EXPERIMENT_NAME=$1
BUCKET_NAME="hamishi-tpu"

# where model will be saved
MODEL_DIR="gs://${BUCKET_NAME}${EXPERIMENT_NAME}/model"
EVAL_OUTPUT_DIR="gs://${BUCKET_NAME}/${EXPERIMENT_NAME}/eval"


# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
  --gin_search_paths=gins \
  --gin_file="hyper_xl.gin" \
  --gin_file="instruction_embed.gin" \
  --gin_file="t0_eval.gin" \
  --gin.hyper_network.HyperT5Config.use_adapter=False \
  --gin.hyper_network.HyperT5Config.use_prefix=False \
  --gin.hyper_network.HyperT5Config.use_instructions=True \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.utils.DatasetConfig.batch_size=128 \
  --gin.USE_CACHED_TASKS=True \
  --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
  --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\" \
  --gin.utils.RestoreCheckpointConfig.mode=\"all\"
