# l'il script for running locally. Make sure to login to gcloud for cached data, or change tfds_data_dir

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m t5x.train \
    --gin_search_paths=./gins \
    --gin_file="t0_train_local.gin" \
    --gin_file="lora/lora_small.gin" \
    --gin.DROPOUT_RATE=0.1 \
    --gin.MODEL_DIR=\"lora_test\" \
    --gin.USE_CACHED_TASKS=False \
    --gin.TRAIN_STEPS=1100010 \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_small/checkpoint_1100000\"
