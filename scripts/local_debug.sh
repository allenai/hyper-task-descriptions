# script for loading existing models locally
# disable jit so I can debug.

EXPERIMENT_NAME=$1

# where model will be saved
MODEL_DIR="gs://hamishi-tpu-bucket/${EXPERIMENT_NAME}/model"

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 JAX_DISABLE_JIT=1 python -m t5x.train \
    --gin_search_paths=./gins \
    --gin_file="t0_train_local.gin" \
    --gin.TRAIN_STEPS=1117000 \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --gin.DROPOUT_RATE=0.1 \
    --gin.USE_CACHED_TASKS=False
