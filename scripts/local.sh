# l'il script for running locally. Make sure to login to gcloud for cached data, or change tfds_data_dir

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 JAX_DISABLE_JIT=1 python -m t5x.train \
    --gin_search_paths=./gins \
    --gin_file="t0_train_local.gin" \
    --gin.MODEL_DIR=\"test\" \
    --gin.USE_CACHED_TASKS=False \
    --gin.TRAIN_STEPS=1100100 \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_small/checkpoint_1100000\"
