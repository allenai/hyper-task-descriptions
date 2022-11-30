# NI training
# for standalone eval

# name of experiment folder
EXPERIMENT_NAME=$1

# where model will be saved
MODEL_DIR="gs://hamishi-us-bucket/${EXPERIMENT_NAME}/model"


EVAL_OUTPUT_DIR="${EXPERIMENT_NAME}/eval/"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m t5x.eval \
    --gin_search_paths="gins" \
    --gin_file="hyper_base.gin" \
    --gin_file="ni_eval.gin" \
    --gin.hyper_network.HyperT5Config.use_adapter=False \
    --gin.hyper_network.HyperT5Config.use_prefix=False \
    --gin.hyper_network.HyperT5Config.use_instructions=False \
    --gin.MIXTURE_OR_TASK_NAME=\"natural_instructions_def\" \
    --gin.USE_CACHED_TASKS=True \
    --gin.utils.DatasetConfig.batch_size=512 \
    --gin.utils.DatasetConfig.split=\"test\" \
    --gin.partitioning.PjitPartitioner.num_partitions=2 \
    --gin.CHECKPOINT_PATH=\"$MODEL_DIR\" \
    --gin.utils.RestoreCheckpointConfig.mode=\"all\" \
    --gin.EVAL_OUTPUT_DIR=\"$EVAL_OUTPUT_DIR\"
