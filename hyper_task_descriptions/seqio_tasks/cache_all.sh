
export PYTHONPATH=/Users/hamishivison/Programming/hyper-task-descriptions

PROJECT=ai2-allennlp
REGION=us-central1
TASK_NAME=seqio-t0-custom
BUCKET=gs://hamishi-dev

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 seqio_cache_tasks \
   --output_cache_dir=gs://hamishi-dev/t0_data_roberta \
   --module_import=hyper_task_descriptions.seqio_tasks.all_t0_tasks \
   --alsologtostderr \
   --pipeline_options="--runner=DataflowRunner,--project=$PROJECT,--region=$REGION,--job_name=$TASK_NAME,--staging_location=$BUCKET/binaries,--temp_location=$BUCKET/tmp,--extra_package=/Users/hamishivison/Programming/seqio/dist/seqio-0.0.7.tar.gz,--requirements_file=beam_requirements.txt,--extra_package=/Users/hamishivison/Programming/hyper-task-descriptions/dist/hyper_task_descriptions-0.1.0.tar.gz"

 
 