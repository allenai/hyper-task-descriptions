# l'il script for running locally. Make sure to login to gcloud for cached data, or change tfds_data_dir

HF_DATASETS_OFFLINE=1 python -m t5x.train \
    --gin_search_paths=./gins \
    --gin_file="t0_train_local.gin" \
    --gin.MODEL_DIR=\"model_test\" \
    --tfds_data_dir="gs://hamishi-tpu-bucket/t0_data/data/" \
