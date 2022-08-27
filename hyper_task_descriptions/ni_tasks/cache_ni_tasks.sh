export PYTHONPATH=/Users/hamishivison/Programming/hyper-task-descriptions

PROJECT=ai2-tpu
REGION=us-central1
BUCKET=gs://hamishi-us-bucket

# offline setting only works if you have 
seqio_cache_tasks \
      --tasks="natural_instructions" \
      --output_cache_dir=gs://hamishi-us-bucket/ni_t5 \
      --module_import=hyper_task_descriptions.ni_tasks.ni_registry \
      --min_shards 1024 \
      --alsologtostderr