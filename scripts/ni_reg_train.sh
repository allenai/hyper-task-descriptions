# name of experiment folder
EXPERIMENT_NAME=$1

# where model will be saved
MODEL_DIR="gs://yizhongw-tpu-bucket/${EXPERIMENT_NAME}/model"

# we go offline to avoid constant calls to get basic info (happens even when cached)
# for your first run, you will probably need to run all these calls :(
python3 -m t5x.train \
  --gin_search_paths=gins \
  --gin_file="hyper_xl.gin" \
  --gin_file="ni_train.gin" \
  --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions_def_pos_2\" \
  --gin.USE_CACHED_TASKS=True \
  --gin.hyper_network.HyperT5Config.add_adapters=False \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.TRAIN_STEPS=1102000 \
  --gin.utils.DatasetConfig.batch_size=1024 \
  --gin.partitioning.PjitPartitioner.num_partitions=8 \
  --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_xl/checkpoint_1100000/\" \
  --tfds_data_dir=\"gs://yizhongw-tpu-bucket/t0_data/data\"


EVAL_OUTPUT_DIR="gs://yizhongw-tpu-bucket/${EXPERIMENT_NAME}/eval/"
python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_xl.gin" \
    --gin_file="ni_eval.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions_def_pos_2\" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=1024 \
    --gin.utils.DatasetConfig.split=\"test\" \
    --gin.partitioning.PjitPartitioner.num_partitions=8 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"all\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"